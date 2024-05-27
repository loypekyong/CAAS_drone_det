from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

results = model.train(data = 'data.yaml', epochs = 3, save = True)

