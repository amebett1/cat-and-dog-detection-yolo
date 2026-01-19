from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

model.predict(source='images.jpg', save=True, conf=0.5, show=True)