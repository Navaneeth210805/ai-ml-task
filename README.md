<h1>TextRegionProcessor: Automated Text Detection from Images</h1>
    <p>
        This project leverages OpenCV and Tesseract OCR to detect and extract text from images, 
        particularly focusing on enhancing word detection, text merging, and preprocessing steps 
        like green line removal and red pixel analysis. The code provides an efficient method to 
        dynamically adjust text region detection and padding while ensuring better OCR accuracy.
    </p>
        <h2>Table of Contents</h2>
    <ul>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#how-it-works">How It Works</a></li>
        <li><a href="#file-descriptions">File Descriptions</a></li>
        <li><a href="#usage">Usage</a></li>
        <li><a href="#result">Result</a></li>
        <li><a href="#limitations">Limitations</a></li>
    </ul>

<h2 id="installation">Installation</h2>
    <h3>Prerequisites</h3>
    <p>Ensure the following libraries are installed in your environment:</p>
    <ul>
        <li>OpenCV (<code>cv2</code>)</li>
        <li>NumPy</li>
        <li>PyTesseract (<code>pytesseract</code>)</li>
        <li>Pandas</li>
        <li>Python (3.6+)</li>
    </ul>

<h3>Setup</h3>
    <p>To install the required Python libraries, run:</p>
    <pre><code>pip install opencv-python numpy pytesseract pandas</code></pre>

<p>You also need to have Tesseract-OCR installed on your system. You can download it 
        <a href="https://github.com/tesseract-ocr/tesseract">here</a>.
    </p>

<h3>Installation of Tesseract-OCR (Linux)</h3>
    <pre><code>sudo apt-get update
    sudo apt-get install tesseract-ocr
    sudo apt-get install libtesseract-dev</code></pre>

<h3>Installation of Tesseract-OCR (Windows)</h3>
    <ol>
        <li>Download Tesseract from the official <a href="https://github.com/tesseract-ocr/tesseract">Tesseract GitHub page</a>.</li>
        <li>Set the path to the Tesseract executable in your environment variables.</li>
    </ol>

<h2 id="how-it-works">How It Works</h2>
    <p>The <code>TextRegionProcessor</code> class extracts and processes text regions from an image using advanced preprocessing techniques like:</p>
    <ol>
        <li><strong>Red Pixel Detection:</strong> Identifies and handles regions containing red text, separating text blocks based on the Y-axis.</li>
        <li><strong>Green Line Removal:</strong> Detects and removes green lines that might interfere with text extraction by inpainting the green regions.</li>
        <li><strong>Adaptive Padding:</strong> Dynamically adjusts padding for text regions based on detected content and boundaries, improving text extraction accuracy.</li>
        <li><strong>Text Merging:</strong> Horizontally merges nearby text regions, ensuring proper word concatenation and removing text region overlaps.</li>
        <li><strong>Validation:</strong> After extracting text, the results are validated and cleaned for alphabetic text only, filtering out erroneous detections.</li>
    </ol>

<h2 id="file-descriptions">File Descriptions</h2>
    <ul>
        <li><code>TextRegionProcessor</code>: The main class that processes the image, detects text, and merges nearby regions.</li>
        <li><strong><code>process_image</code>:</strong> Processes an image and extracts text, using dynamic padding to improve the accuracy of OCR detection.</li>
        <li><strong><code>merge_nearby_regions</code>:</strong> Merges close horizontal text regions to avoid fragmentation.</li>
        <li><strong><code>detect_red_pixels_with_y_distance</code>:</strong> Handles special cases where red text needs to be separated based on vertical pixel distance.</li>
        <li><strong><code>remove_green_lines</code>:</strong> Detects green lines and removes them by inpainting.</li>
        <li><strong><code>validate_results</code>:</strong> Filters OCR results to include only valid alphabetic characters.</li>
        <li><strong><code>extract_text_from_region</code>:</strong> Extracts text from regions with enhanced preprocessing and confidence checking.</li>
        <li><strong><code>main.py</code>:</strong> The main file that uses <code>TextRegionProcessor</code> to process an image and extract validated text regions.</li>
    </ul>

<h2 id="usage">Usage</h2>
    <h3>Running the Script</h3>
    <p>
        1. <strong>Place your image</strong> (e.g., <code>sample.jpeg</code>) in the appropriate directory.<br>
        2. Run the script using Python:
    </p>
    <pre><code>python main.py</code></pre>

<p>The output will be generated in a specified directory (e.g., <code>/kaggle/working/augmented_images/</code>) with the following files:</p>
    <ul>
        <li><strong>annotated_image.png:</strong> The processed image with bounding boxes drawn around the detected text regions.</li>
        <li><strong>final_result.json:</strong> The extracted and cleaned text data in JSON format.</li>
    </ul>

<h3>Example Code</h3>
    <pre><code>processor = TextRegionProcessor(image_path='path/to/image.jpeg', save_dir='output_directory/')
results = processor.process_image()
validated_results = processor.validate_results(results)

# Print validated results
print(validated_results)
</code></pre>

<h3>Output</h3>
    <p>The extracted and validated text regions will be saved in <code>final_result.json</code>. Each key corresponds to a detected text region, and the values represent the text found in those regions.</p>

<h2 id="result">Result</h2>
    <p>The processed image will have the detected text regions outlined, and the extracted text will be saved as a JSON file (<code>final_result.json</code>). Example output format in JSON:</p>
    <pre><code>[
    {
        "region": "Hypothalamus",
        "text": "TRH, CRH, GHRH Dopamine"
    },
    {
        "region": "Stomach",
        "text": "Gastrin, Ghrelin, Histamine"
    }
]
</code></pre>

<h2 id="limitations">Limitations</h2>
    <ol>
        <li><strong>Red Pixel Detection:</strong> The red pixel detection works based on fixed thresholds, which may not generalize to all images.</li>
        <li><strong>Region Merging:</strong> Sometimes, region merging might group text incorrectly if padding values are too large or too small.</li>
        <li><strong>Small Font Size:</strong> The script might struggle with very small font sizes, as region detection requires a minimum contour size.</li>
        <li><strong>Language Support:</strong> Currently optimized for English alphabetic characters; can be extended with additional language support in Tesseract.</li>
    </ol>

<p>By following this documentation, you should be able to run the script, process images, and extract meaningful text data with minimal effort. Feel free to adjust parameters within the <code>TextRegionProcessor</code> to suit your specific image and text detection needs.</p>
