from pathlib import Path
from ultralytics import YOLO

DATASET = "hand-keypoints.yaml"

MODELS = {
    "1": "yolo11n-pose.pt",
    "2": "yolo11s-pose.pt",
    "3": "yolo26n-pose.pt",
    "4": "yolo26s-pose.pt",
}

EPOCHS = 50
IMG_SIZE = 640
BATCH = 64
DEVICE = 0
WORKERS = 0
PROJECT = "runs_pose"


def train_model(model_name):
    print("=" * 60)
    print(f"Training {model_name}")
    print("=" * 60)

    model = YOLO(model_name)

    model.train(
        data=DATASET,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        name=Path(model_name).stem,
        pretrained=True,
        cache=False,
        save=True,
        verbose=True
    )

    print("=" * 60)
    print(f"\nDone {model_name}\n")
    print("=" * 60)

def choice_model():

    print("=" * 60)
    print("1. YOLO11n-pose")
    print("2. YOLO11s-pose")
    print("3. YOLO26n-pose")
    print("4. YOLO26s-pose")
    print("=" * 60)
     
    choice = input("Choice model: ").strip()
    
    if choice not in MODELS:
        print("Skip training.")
        return

    return choice

def main():

    choice = choice_model()

    if choice in MODELS:
        model = MODELS[choice]
        train_model(model)
    

if __name__ == "__main__":
    main()