import json
from ultralytics import YOLO

# Load the model
model = YOLO("model.pt")

# Run validation on test split
metrics = model.val(data="data.yaml", split="val")

# Extract metrics dictionary
metrics_dict = metrics.results_dict

# Save to JSON file
with open("test_metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)

# Optional: Save a pretty-printed text version
with open("test_metrics.txt", "w") as f:
    for k, v in metrics_dict.items():
        f.write(f"{k}: {v}\n")