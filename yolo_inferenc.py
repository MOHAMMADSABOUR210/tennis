from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.predict('input/input_video.mp4',conf=0.2, save=True)

print(result)

print("boxes: ")

for box in result[0].boxes:
    print(box)

# D:\Programming\Artificial intelligence\project\tenis_yolo\training
# models\best.pt