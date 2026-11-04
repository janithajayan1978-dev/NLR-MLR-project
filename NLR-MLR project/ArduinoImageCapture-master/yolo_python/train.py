
from pathlib import Path
from ultralytics import YOLO
import yaml

# Dataset configuration
dataset_dir = Path.home() / "Downloads" / "dataset-yolo" / "dataset-yolo"
source_yaml = dataset_dir / "data.yaml"

# Read the dataset configuration
with open(source_yaml, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Use absolute paths so training can find the images
config["path"] = str(dataset_dir)
config["train"] = "train/images"
config["val"] = "valid/images"
config["test"] = "test/images"

# Save a corrected configuration inside yolo_python
output_yaml = Path(__file__).parent / "data.yaml"
with open(output_yaml, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

# Start training
model = YOLO("yolo11n.pt")

model.train(
    data=str(output_yaml),
    epochs=10,
    imgsz=320,
    batch=1,
    workers=0,
    cache=False,
    plots=False,
    project=str(Path(__file__).parent / "runs"),
    name="blood_cells",
    exist_ok=True
)
