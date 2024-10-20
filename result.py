# JSON data as input
import json

# Load JSON data from a file
with open('./final_result.json', 'r') as f:
    json_data = json.load(f)
# Headings extracted from the image
headings = [
    "Hypothalamus", "Thyroid and Parathyroid", "Liver", "Adrenal", "Kidney", 
    "Testes", "Ovary and Placenta", "Pineal Gland", "Pituitary gland", "Thymus",
    "Stomach", "Pancreas", "Uterus"
]


# Create a structured dictionary where each heading will have its respective data entries.
structured_data = {heading: [] for heading in headings}

# Function to check if an entry is a known heading
def is_heading(entry):
    return entry in headings or entry in ["Ovary, Placenta", "Thyroid and"]

# Process the JSON data
for entry in json_data:
    if is_heading(entry["0"]):
        if entry["0"] == "Ovary, Placenta":
            key = "Ovary and Placenta"
        elif entry["0"] == "Thyroid and":
            key = "Thyroid and Parathyroid"
        else:
            key = entry["0"]
        for i in range(1, 5):
            value = entry.get(f"{i}")
            if value and value not in structured_data[key]:
                structured_data[key].append(value)

structured_data = {k: [v for v in vals if v] for k, vals in structured_data.items()}

for heading, sub_entries in structured_data.items():
    print(f"{heading}:")
    for sub_entry in sub_entries:
        print(f"  - {sub_entry}")
    print()