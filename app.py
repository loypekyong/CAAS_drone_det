import glob
import streamlit as st
from PIL import Image
import torch
import cv2
import os
import time
from ultralytics import YOLO
import supervision as sv
import numpy as np

# torch.cuda.set_device(0)
# import the inference-sdk
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (255, 255, 255)  # White color in BGR
thickness = 2
position = (10,30)
# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="YDDPJSRTg55MWHqiwYKy"
)

st.set_page_config(layout="wide")

model = None
confidence = .25
model_paths = {
    "YOLOv9": "best (unresumed).pt",
    "Roboflow": "Roboflow",
    "Model3": "Path3"
    }
#t


def image_input():
    img_file = None
    img_bytes = st.sidebar.file_uploader("Upload an image", type=['png','jpeg', 'jpg'])

    if img_bytes:
        img_file = "Stuff/upload_image."+ img_bytes.name.split('.')[-1]
        Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            infer_image(img_file)
            st.image("Stuff/result.jpg", caption="Model prediction")
            
            

def video_input():
    vid_file = None
    vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
    if vid_bytes:
        vid_file = "Stuff/upload_vid." + vid_bytes.name.split('.')[-1]
        with open(vid_file, 'wb') as out:
            out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)
        
        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        out = cv2.VideoWriter('Stuff/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                out.release()
                break
            frame = cv2.resize(frame, (width, height))
            # frame = cv2.bitwise_not(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # st.image(frame)
            infer_image(frame)
            output.image("Stuff/result.jpg")
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")
            out.write(np.array(Image.open("Stuff/result.jpg"))[:, :, ::-1])

        cap.release()
        

def infer_image(img, size=None):
    if selected_model_name == "YOLOv9":
        results = model(img, size=size, classes=classes, conf = confidence) if size else model(img, classes=classes, conf = confidence)
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            # result.show()  # display to screen
            result.save(filename='Stuff/result.jpg')
    elif selected_model_name == "Roboflow":
        if input_option == "image":
            image = Image.open("Stuff/upload_image.jpg")
        else:
            image = img
        custom_configuration = InferenceConfiguration(confidence_threshold=confidence)
        CLIENT.configure(custom_configuration)
        results = CLIENT.infer(image, model_id="caas-drone-detection/1")
        detections = sv.Detections.from_inference(results)
        detections = detections[np.isin(detections.class_id, classes)]
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
        if input_option == "image":
            annotated_image.save("Stuff/result.jpg")
        else:
            cv2.imwrite("Stuff/result.jpg", annotated_image)

        


@st.cache_resource
def load_model(selected_model_name):
    if selected_model_name == "YOLOv9":
        model = YOLO(model_paths[selected_model_name])
        # model.export(format='onnx')

    else: model = selected_model_name
    return model


def main():
    # global variables
    global model, confidence, selected_model_name, classes, input_option

    st.title("CAAS drone detection")

    st.sidebar.title("Settings")

    selected_model_name = st.sidebar.selectbox("Select a model", list(model_paths.keys()))
    st.header("Selected Model is: " + selected_model_name)

    if selected_model_name == "YOLOv9":
        model = load_model(selected_model_name)
        

    # confidence slider
    confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)
    model_names = ["aircraft", "bird", "drone"]

    assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[2]])
    classes = [model_names.index(name) for name in assigned_class]  

    st.sidebar.markdown("---")

    # input options
    input_option = st.sidebar.radio("Select input type: ", ['image', 'video'])
    
    if input_option == 'image':
        # image_input(data_src)
        image_input()
    else:
        # video_input(data_src)
        video_input()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass