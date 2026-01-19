from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

model.predict(source='test_images.jpg', save=True, conf=0.5, show=True)