import torch
import cv2
import numpy as np
import supervision as sv

import threading
import emoji
from transformers import (
    AutoImageProcessor, 
    AutoModelForObjectDetection
)


def just_detection(input):
    print("Just Only",input)
    return input



just_detection("car")


def objek_deteksi():
    CHECKPOINT = "PekingU/rtdetr_r50vd_coco_o365"

    DEVICE = torch.device("cuda")
    if DEVICE.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    model = AutoModelForObjectDetection.from_pretrained(CHECKPOINT).to(DEVICE)
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT, use_fast=True)

    video_path = "beach.mp4"
    cap = cv2.VideoCapture(video_path)
    frame_skip = 3  # Process every other frame to speed up video
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames to speed up video

        frame = cv2.resize(frame, (1280, 720))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = processor(rgb_frame, return_tensors="pt").to(DEVICE)   

        with torch.no_grad():
            outputs = model(**inputs)

        h, w = frame.shape[:2]
        results = processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=0.5)

        detections = sv.Detections.from_transformers(results[0])
        labels = [
            model.config.id2label[class_id]
            for class_id
            in detections.class_id
        ]

        annotated_frame = sv.BoxAnnotator().annotate(frame, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels=labels)

        cv2.imshow('Object Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    objek_deteksi()