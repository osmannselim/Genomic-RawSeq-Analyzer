import json
import base64
import os

notebook_path = "notebooks/EvalDNABERT2.ipynb"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

img_count = 0
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        for output in cell.get("outputs", []):
            if "data" in output and "image/png" in output["data"]:
                img_data = output["data"]["image/png"]
                img_bytes = base64.b64decode(img_data)
                
                # If there are multiple images, name them accordingly
                img_count += 1
                out_path = os.path.join(output_dir, f"dnabert2_evaluation_plots_{img_count}.png" if img_count > 1 else "dnabert2_evaluation_plots.png")
                
                with open(out_path, "wb") as img_file:
                    img_file.write(img_bytes)
                print(f"Successfully extracted image to: {out_path}")

if img_count == 0:
    print("No PNG images found in the notebook outputs.")
