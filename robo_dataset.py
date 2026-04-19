# !pip install -q ultralytics roboflow
from roboflow import Roboflow
from ultralytics import YOLO
import torch
import os
                
if __name__ == '__main__':
    rf = Roboflow(api_key="kWhhzcsndL8DHL6oYFfF")
    project = rf.workspace("final-year-project-yiiyi").project("indian-currency-coin-detection")
    version = project.version(5)
    dataset = version.download("yolo26")
    os.environ["DATASET_LOCATION"] = dataset.location
    print(f"Dataset downloaded to: {dataset.location}")


    # 2. Load the YOLO model
    # 'yolo11n.pt' is the fastest/lightest; use 's', 'm', 'l', or 'x' for more accuracy
    model = YOLO("yolo26n.pt") 

    # 3. Train with High GPU Utilization settings
    results = model.train(
        data=f"{dataset.location}/data.yaml", # Path to the downloaded dataset
        epochs=100,                           # Number of training loops
        imgsz=640,                            # Image size
        batch=-1,                             # AUTO-BATCH: Finds max size for your GPU VRAM
        workers=8,                            # Parallel threads for data loading (adjust based on CPU)
        device=0,                             # Use GPU '0'. Use [0, 1] for multi-GPU
        amp=True,                             # Automatic Mixed Precision (faster training)
        plots=True                            # Save progress charts
    )