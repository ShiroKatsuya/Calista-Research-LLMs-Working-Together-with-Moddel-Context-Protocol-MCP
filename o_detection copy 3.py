import torch
import cv2
import numpy as np
import supervision as sv

import threading
import emoji
from transformers import AutoImageProcessor, AutoModelForObjectDetection

def just_detection(input):
    print("Just Only", input)
    return input


# filter_object = just_detection(["boat", "person"])
# related = ["boat", "person"]

filter_object = just_detection("boat")
related = ["boat"]

def objek_deteksi():
    CHECKPOINT = "facebook/detr-resnet-50"

    DEVICE = torch.device("cuda")
    if DEVICE.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    model = AutoModelForObjectDetection.from_pretrained(CHECKPOINT).to(DEVICE)
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT, use_fast=True)

    video_path = "boat.mp4"
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
            outputs, target_sizes=[(h, w)], threshold=0.7)

        detections = sv.Detections.from_transformers(results[0])
        labels = [
            model.config.id2label[class_id]
            for class_id
            in detections.class_id
        ]
        
        print(f"Detected objects: {labels}")
        
        # Handle both single string and list of strings for filter_object
        filter_list = filter_object if isinstance(filter_object, list) else [filter_object]
        print(f"Looking for: {filter_list}")
        
        # Filter detections based on the filter_object
        filtered_indices = []
        filtered_labels = []
        for i, label in enumerate(labels):
            label_lower = label.lower()
            
            # Check if the label matches any of our filter objects
            for filter_item in filter_list:
                filter_item_lower = filter_item.lower() if isinstance(filter_item, str) else ""
                
                # If filter_item is "car", use the list of car-related terms
                if filter_item_lower == "car" and any(term in label_lower for term in related):
                    filtered_indices.append(i)
                    filtered_labels.append(label)
                    break
                # Otherwise do exact matching for other object types
                elif filter_item_lower == label_lower:
                    filtered_indices.append(i)
                    filtered_labels.append(label)
                    break
        
        print(f"Filtered objects: {filtered_labels}")
        
        if filtered_indices:  # Only keep detections that match the filter
            detections = detections[filtered_indices]
            labels = filtered_labels
            
            # Only display annotations if we have filtered detections
            annotated_frame = sv.BoxAnnotator().annotate(frame, detections)
            annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels=labels)
        else:
            # No matching objects found, just show the original frame
            annotated_frame = frame

        cv2.imshow('Object Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    objek_deteksi()