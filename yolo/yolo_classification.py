from ultralytics import YOLO
import os

def classify_image(model_path: str, image_path: str):
    """
    Classify an image using a YOLO model.

    Args:
        model_path (str): Path to the YOLO model file.
        image_path (str): Path to the image file to classify.

    Returns:
        List[dict]: Detected objects with class name, confidence, and bounding box.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Load the YOLO model
    model = YOLO(model_path)
    results = model(image_path)

    detections = []
    for result in results:
        names = model.names  # Map class indices to names
        for box in result.boxes:
            class_id = int(box.cls[0])  # Convert from tensor to int
            detections.append({
                'class': names[class_id],
                'confidence': round(float(box.conf[0]), 4),
                'bbox': [round(coord, 2) for coord in box.xyxy[0].tolist()]  # Format nicely
            })

    return detections


if __name__ == "__main__":
    model_path = "./model.pt"
    image_path = "./dataset/images/test/Ripe_Mango_0_3031.jpg"

    try:
        detections = classify_image(model_path, image_path)
        print("Detections:")
        for det in detections:
            print(f" - {det['class']} ({det['confidence']*100:.1f}%): {det['bbox']}")
    except FileNotFoundError as e:
        print(f"[Error] {e}")