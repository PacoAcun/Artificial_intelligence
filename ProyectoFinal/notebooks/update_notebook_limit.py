import json
import os

notebook_path = '06_Pipeline_YOLO_Hybrid.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if 'test_images = sorted(test_images)[:5]' in line:
                    # Check if it's exactly the line we want to change (ignoring potential trailing newlines in the list)
                    # The line in json might be "test_images = sorted(test_images)[:5]\n"
                    if 'test_images = sorted(test_images)[:5]' in line: 
                        new_line = line.replace('[:5]', '[:50]')
                        new_source.append(new_line)
                        found = True
                        print(f"Found and replacing line: {line.strip()} -> {new_line.strip()}")
                    else:
                        new_source.append(line)
                else:
                    new_source.append(line)
            cell['source'] = new_source

    if found:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook updated successfully.")
    else:
        print("Target line not found.")

except Exception as e:
    print(f"An error occurred: {e}")
