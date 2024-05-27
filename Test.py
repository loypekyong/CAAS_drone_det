from ultralytics import YOLO
import cv2

model = YOLO('best (unresumed).pt')

video_capture = cv2.VideoCapture("2 Second Video.mp4")
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

results = model.predict("2 Second Video.mp4")
# Create a video writer to save the results
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
# Open the input video

# Iterate through each frame and draw the detected objects
for frame in results:
    ret, image = video_capture.read()
    if not ret:
        break

    for box in frame:
        xmin, ymin, xmax, ymax, conf, cls = box
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f"{model.names[int(cls)]} {conf:.2f}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(image)

# Release video capture and writer
video_capture.release()
out.release()
# # results = model.val()

# results = model(data = 'data.yaml', split = "test")

# # for result in results:
# #     boxes = result.boxes  # Boxes object for bounding box outputs
# #     masks = result.masks  # Masks object for segmentation masks outputs
# #     keypoints = result.keypoints  # Keypoints object for pose outputs
# #     probs = result.probs  # Probs object for classification outputs
# #     obb = result.obb  # Oriented boxes object for OBB outputs
# #     result.show()  # display to screen
