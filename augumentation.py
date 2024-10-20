import cv2
import numpy as np
import os
import pytesseract
from typing import Optional, Dict, List, Tuple
import pandas as pd
import re

class TextRegionProcessor:
    def is_alpha_text(self, text: str) -> bool:
        """
        Check if the text contains only alphabetic characters or common punctuation.
        """
        return bool(re.match(r'^[a-zA-Z\s,.!?\'"]+$', text))

    def __init__(self, image_path: str, save_dir: str):
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Modified configuration for better word merging
        self.config = {
            'min_confidence': 30,
            'min_text_length': 1,
            'padding_x': 200,    # Horizontal padding
            'padding_y': 20,     # Vertical padding
            'merge_threshold': 150,  # Increased threshold for merging nearby regions
            'vertical_merge_threshold': 100,# Threshold for vertical alignment
            'red_text_threshold': 50 
        }

    def detect_red_pixels_with_y_distance(self, image: np.ndarray) -> bool:
        """
        Detect if red pixels are separated by at least 50 pixels on the y-axis and stop processing if found.
        """
        # Convert image to HSV color space to easily detect red pixels
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for red color in HSV (two ranges for red in HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)

        # Combine masks
        red_mask = red_mask1 + red_mask2

        # Loop through each column of the image and check if red pixels are separated by 50 pixels on the y-axis
        red_pixel_y_positions = []
        for y in range(red_mask.shape[0]):  # Iterate over height (y-axis)
            for x in range(red_mask.shape[1]):  # Iterate over width (x-axis)
                if red_mask[y, x] > 0:  # Check if the pixel is red
                    red_pixel_y_positions.append(y)

            # Check if there are any red pixels with at least 50 pixels apart in the y-axis
            for i in range(1, len(red_pixel_y_positions)):
                if abs(red_pixel_y_positions[i] - red_pixel_y_positions[i-1]) >= 20:
                    return True

        return False

    def remove_green_lines(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and remove green lines by filling them with surrounding pixels.
        """
        # Convert image to HSV color space to easily detect green lines
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the range for green color in HSV (adjust these ranges if necessary)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Create a mask for green color
        green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

        # Dilate the mask slightly to cover the entire width of the green lines
        kernel = np.ones((5, 5), np.uint8)
        green_mask_dilated = cv2.dilate(green_mask, kernel, iterations=1)

        # Inpaint the image to remove the green lines and replace them with surrounding pixels
        inpainted_img = cv2.inpaint(image, green_mask_dilated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return inpainted_img

    def merge_nearby_regions(self, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Merge regions with improved horizontal word detection
        """
        if not regions:
            return regions
            
        # Sort regions primarily by y-coordinate (vertical position) and secondarily by x-coordinate
        regions = sorted(regions, key=lambda x: (x[1], x[0]))
        merged = []
        current = list(regions[0])
        
        for next_region in regions[1:]:
            # Check if regions are on the same line (similar y-coordinate)
            same_line = abs((current[1] + current[3]/2) - (next_region[1] + next_region[3]/2)) < self.config['vertical_merge_threshold']
            
            # Check horizontal distance between regions
            horizontal_gap = next_region[0] - (current[0] + current[2])
            
            # If regions are on the same line and close enough horizontally
            if same_line and horizontal_gap < self.config['merge_threshold']:
                # Merge regions
                current[2] = next_region[0] + next_region[2] - current[0]  # Extend width
                current[3] = max(current[3], next_region[3])  # Take max height
            else:
                merged.append(tuple(current))
                current = list(next_region)
                
        merged.append(tuple(current))
        return merged

    def process_image(self) -> Dict[str, str]:
        """
        Process image with improved region merging and dynamic right padding.
        """
        # Remove green lines before text detection
        cleaned_img = self.remove_green_lines(self.img)

        # Convert to grayscale
        gray = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding with modified parameters
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 7, 2  # Adjusted parameters
        )

        # Perform dilation to connect nearby text components
        kernel = np.ones((4, 3), np.uint8)  # Horizontal kernel
        dilated = cv2.dilate(thresh, kernel, iterations=3)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get regions with a minimum area threshold
        regions = [cv2.boundingRect(c) for c in contours if 10000 > cv2.contourArea(c) > 30]

        # Merge nearby regions
        merged_regions = self.merge_nearby_regions(regions)

        result = {}

        for i, (x, y, w, h) in enumerate(merged_regions):
            # Apply initial padding
            padding_2_x = 30
            padding_2_y = 30
            padding_1_y = 30
            x1 = max(0, x - padding_2_x)
            y1 = max(0, y - padding_1_y)
            y2 = min(self.img.shape[0], y + h + padding_2_y)

            # Start with initial right padding
            padding_x = 20
            x2 = min(self.img.shape[1], x + w + padding_x)
            detected_text = ''
            attempts_with_no_new_text = 0

            # Dynamically increase the right padding
            while attempts_with_no_new_text < 3:
                x2 = min(self.img.shape[1], x + w + padding_x)

                # Extract region with current padding
                region = self.img[y1:y2, x1:x2]

                # Skip if region is too small
                if region.shape[0] < 8 or region.shape[1] < 8:
                    break

                # Extract text from the region
                region = self.remove_green_lines(region)
                if self.detect_red_pixels_with_y_distance(region):
                    padding_x -= 30
                    break
                new_text = self.extract_text_from_region(region)
                print("Detected vs new in padding_x", new_text, detected_text)

                # If no new text is detected, increment the counter and stop after 5 failed attempts
                if new_text == detected_text:
                    attempts_with_no_new_text += 1
                else:
                    detected_text = new_text  # Update the detected text
                    attempts_with_no_new_text = 0  # Reset counter if new text is found

                # Increase the right padding for the next iteration
                padding_x += 30

            detected_text = ''
            attempts_with_no_new_text = 0

            while attempts_with_no_new_text < 5:
                x1 = max(0, x - padding_2_x)

                # Extract region with current padding
                region = self.img[y1:y2, x1:x2]

                # Skip if region is too small
                if region.shape[0] < 8 or region.shape[1] < 8:
                    break

                # Extract text from the region
                # Stop if more than 2 red texts are found separated by a considerable distance
                if self.detect_red_pixels_with_y_distance(region):
                    padding_2_x -= 30
                    break
                region = self.remove_green_lines(region)
                new_text = self.extract_text_from_region(region)
                print("Detected vs new in padding_2_x", new_text, detected_text)

                # If no new text is detected, increment the counter and stop after 5 failed attempts
                if new_text == detected_text:
                    attempts_with_no_new_text += 1
                else:
                    detected_text = new_text  # Update the detected text
                    attempts_with_no_new_text = 0  # Reset counter if new text is found

                # Increase the right padding for the next iteration
                padding_2_x += 30

            detected_text = ''
            attempts_with_no_new_text = 0
            max_attempts = 0
            while attempts_with_no_new_text < 2 and max_attempts < 18:
                y2 = min(self.img.shape[0], y + h + padding_2_y)

                # Extract region with current padding
                region = self.img[y1:y2, x1:x2]

                # Skip if region is too small
                if region.shape[0] < 8 or region.shape[1] < 8:
                    break

                # Extract text from the region
                region = self.remove_green_lines(region)
                if self.detect_red_pixels_with_y_distance(region):
                    padding_2_y -= 40
                    break
                new_text = self.extract_text_from_region(region)
                print("Detected vs new in padding_2_y", new_text, detected_text)

                # If no new text is detected, increment the counter and stop after 5 failed attempts
                if new_text == detected_text:
                    attempts_with_no_new_text += 1
                else:
                    detected_text = new_text  # Update the detected text
                    attempts_with_no_new_text = 0  # Reset counter if new text is found

                # Increase the right padding for the next iteration
                max_attempts += 1
                padding_2_y += 40

            detected_text = ''
            attempts_with_no_new_text = 0
            max_attempts = 0
            while attempts_with_no_new_text < 3 and max_attempts < 18:
                y1 = max(0, y - padding_1_y)

                # Extract region with current padding
                region = self.img[y1:y2, x1:x2]

                # Skip if region is too small
                if region.shape[0] < 8 or region.shape[1] < 8:
                    break

                # Extract text from the region
                region = self.remove_green_lines(region)
                if self.detect_red_pixels_with_y_distance(region):
                    padding_1_y -= 40
                    break
                new_text = self.extract_text_from_region(region)
                print("Detected vs new in padding_1_y", new_text, detected_text)

                # If no new text is detected, increment the counter and stop after 5 failed attempts
                if new_text == detected_text:
                    attempts_with_no_new_text += 1
                else:
                    detected_text = new_text  # Update the detected text
                    attempts_with_no_new_text = 0  # Reset counter if new text is found

                # Increase the right padding for the next iteration
                max_attempts += 1
                padding_1_y += 40
            detected_text = ''
            attempts_with_no_new_text = 0
            while attempts_with_no_new_text < 3:
                x2 = min(self.img.shape[1], x + w + padding_x)

                # Extract region with current padding
                region = self.img[y1:y2, x1:x2]

                # Skip if region is too small
                if region.shape[0] < 8 or region.shape[1] < 8:
                    break

                # Extract text from the region
                region = self.remove_green_lines(region)
                if self.detect_red_pixels_with_y_distance(region):
                    padding_x -= 30
                    break
                new_text = self.extract_text_from_region(region)
                print("Detected vs new in padding_x", new_text, detected_text)

                # If no new text is detected, increment the counter and stop after 5 failed attempts
                if new_text == detected_text:
                    attempts_with_no_new_text += 1
                else:
                    detected_text = new_text  # Update the detected text
                    attempts_with_no_new_text = 0  # Reset counter if new text is found

                # Increase the right padding for the next iteration
                padding_x += 30

            detected_text = ''
            attempts_with_no_new_text = 0

            while attempts_with_no_new_text < 5:
                x1 = max(0, x - padding_2_x)

                # Extract region with current padding
                region = self.img[y1:y2, x1:x2]

                # Skip if region is too small
                if region.shape[0] < 8 or region.shape[1] < 8:
                    break

                # Extract text from the region
                region = self.remove_green_lines(region)
                if self.detect_red_pixels_with_y_distance(region):
                    padding_2_x -= 30
                    break
                new_text = self.extract_text_from_region(region)
                print("Detected vs new in padding_2_x", new_text, detected_text)

                # If no new text is detected, increment the counter and stop after 5 failed attempts
                if new_text == detected_text:
                    attempts_with_no_new_text += 1
                else:
                    detected_text = new_text  # Update the detected text
                    attempts_with_no_new_text = 0  # Reset counter if new text is found

                # Increase the right padding for the next iteration
                padding_2_x += 30
            # Store the detected text if any
            if detected_text:
                key = f"text_region_{i}"
                result[key] = detected_text

                # Save region for debugging
                cv2.imwrite(os.path.join(self.save_dir, f"{key}.png"), region)

                # Draw rectangles on the original image (for visualizing)
                cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save annotated image
        cv2.imwrite(os.path.join(self.save_dir, "annotated_image.png"), self.img)

        return result if result else {"error": "No valid text detected"}

    def extract_text_from_region(self, region: np.ndarray) -> Optional[str]:
        """Extract text from region using OCR with alpha-only filter."""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.fastNlMeansDenoising(thresh)

        custom_config = r'--oem 1 --psm 3'
        ocr_data = pytesseract.image_to_data(denoised, output_type=pytesseract.Output.DICT, config=custom_config)

        text_lines = []
        current_line = []
        last_line_num = -1

        for i, (word, conf, line_num) in enumerate(zip(ocr_data['text'], ocr_data['conf'], ocr_data['line_num'])):
            word = word.strip()
            if conf > self.config['min_confidence'] and word and self.is_alpha_text(word):
                if line_num != last_line_num:
                    if current_line:
                        text_lines.append(current_line)
                        current_line = []
                    last_line_num = line_num
                current_line.append(word)

        if current_line:
            text_lines.append(current_line)

        return text_lines if text_lines else None


    def validate_results(self, results: Dict[str, List[List[str]]]) -> Dict[str, List[str]]:
        """Validate and clean up the detected text results."""
        cleaned_results = {}
        
        for key, lines in results.items():
            if lines and isinstance(lines, list):
                cleaned_lines = [
                    ' '.join(line).strip() 
                    for line in lines 
                    if line and len(' '.join(line).strip()) > 3 and all(self.is_alpha_text(word) for word in line)
                ]
            
                if cleaned_lines:
                    cleaned_results[key] = cleaned_lines
        
        return cleaned_results if cleaned_results else {"error": "No valid text after cleaning"}


import pandas as pd
import json

def main():
    try:
        # Change the output directory to /kaggle/working/ which is writable
        output_dir = '/kaggle/working/augmented_images/'
        os.makedirs(output_dir, exist_ok=True)
        
        processor = TextRegionProcessor(
            image_path='/kaggle/input/ai-ml-task/sample.jpeg',
            save_dir=output_dir
        )

        results = processor.process_image()
        validated_results = processor.validate_results(results)
        print(validated_results)

        final_result = list(set(tuple(v) for v in validated_results.values()))
        final_result2 = []
        for i in range(len(final_result)):
            x = []
            for j in range(len(final_result[i])):
                if final_result[i][j][:2] == "ee":
                    break
                if final_result[i][j] in x:
                    continue
                x.append(final_result[i][j])
            if len(x) < 2:
                continue
            final_result2.append(x)

        df1 = pd.DataFrame(final_result2)
        df1_json = df1.to_json(orient="records", indent=4)

        # Change the output path to /kaggle/working/
        output_json_path = "/kaggle/working/final_result.json"
        
        with open(output_json_path, "w") as json_file:
            json_file.write(df1_json)

        with open(output_json_path, "r") as json_file:
            for line in json_file:
                print(line.strip())

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

if __name__ == "__main__":
    main()