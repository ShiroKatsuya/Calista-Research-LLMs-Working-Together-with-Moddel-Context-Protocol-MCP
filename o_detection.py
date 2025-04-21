import torch
import cv2
import numpy as np
import supervision as sv

import threading
import emoji
import time
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from colution import calculate_destruction_probability
from zoom import apply_zoom
def just_detection(input):
    print("Just Only", input)
    return input


# filter_object = just_detection(["boat", "person"])
# related = ["boat", "person"]

filter_object = just_detection("person")
related = ["person"]



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
    frame_skip = 3  
    frame_count = 0
    
    # Enhanced zoom functionality variables
    is_zoomed = False
    zoom_target = None
    zoom_scale = 2.0  # Zoom scale factor
    current_zoom_level = 1.0  # Start with no zoom
    target_zoom_level = 1.0
    zoom_speed = 0.1  # Control how fast zoom transitions happen
    current_zoom_object_id = None
    multi_object_mode = False
    high_prob_targets = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue 

        original_frame = frame.copy()
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

        all_labels = labels
        
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
            enhanced_labels = []
            high_prob_objects = []
            
            for i, label in enumerate(labels):
                prob = calculate_destruction_probability(detections[i:i+1], frame.shape)
                enhanced_labels.append(f"{label}-Person Help: {prob:.0%}")
                
                # Track all objects with their probability
                high_prob_objects.append((i, prob, detections[i]))
            
            # # Sort all objects by probability (highest first)
            high_prob_objects.sort(key=lambda x: x[1], reverse=True)
            
            # # Check for multiple objects with similar probabilities (within 5% of each other)
            similar_high_prob_objects = []
            if high_prob_objects and high_prob_objects[0][1] >= 0.7:
                top_prob = high_prob_objects[0][1]
                for idx, prob, detection in high_prob_objects:
                    # Consider objects with probability within 5% of the top object
                    if abs(prob - top_prob) <= 0.05 and prob >= 0.7:
                        similar_high_prob_objects.append((idx, prob, detection))
            
            # # Multi-object mode when we have multiple objects with similar high probabilities
            if len(similar_high_prob_objects) > 1:
                multi_object_mode = True
                high_prob_targets = similar_high_prob_objects
                target_zoom_level = 1.5  # Partial zoom to show multiple objects
                # We'll send all targets to the zoom function
                zoom_target = [det for _, _, det in similar_high_prob_objects]
                current_zoom_object_id = None  # No single target in multi-object mode
            else:
                multi_object_mode = False
                high_prob_targets = []
                
                # Default single-object behavior
                current_target_valid = False
                
                # Check if current zoom target still has ≥70% probability
                if current_zoom_object_id is not None:
                    for idx, prob, detection in high_prob_objects:
                        if idx == current_zoom_object_id and prob >= 0.7:
                            current_target_valid = True
                            zoom_target = detection
                            target_zoom_level = 2.0  # Full zoom
                            break
                
            #     # If current target is not valid (below 70% or not found)
                if not current_target_valid:
                    # Find a new target with ≥70% probability
                    new_target_found = False
                    for idx, prob, detection in high_prob_objects:
                        if prob >= 0.7:
                            current_zoom_object_id = idx
                            zoom_target = detection
                            target_zoom_level = 2.0  # Full zoom
                            new_target_found = True
                            break
                    
                    # If no high probability objects found, zoom out
                    if not new_target_found:
                        current_zoom_object_id = None
                        zoom_target = None
                        target_zoom_level = 1.0  # No zoom
            
            # Smoothly adjust zoom level
            if abs(current_zoom_level - target_zoom_level) > 0.01:
                if current_zoom_level < target_zoom_level:
                    current_zoom_level = min(current_zoom_level + zoom_speed, target_zoom_level)
                else:
                    current_zoom_level = max(current_zoom_level - zoom_speed, target_zoom_level)
            
            # Apply zoom using the dedicated function
            if multi_object_mode:
                annotated_frame = apply_zoom(frame, zoom_target, current_zoom_level, detections, enhanced_labels, multi_object_mode=True)
            else:
                annotated_frame = apply_zoom(frame, zoom_target, current_zoom_level, detections, enhanced_labels)

            # Add info text about currently tracked object
            # if current_zoom_object_id is not None and high_prob_objects and not multi_object_mode:
                for idx, prob, _ in high_prob_objects:
                    if idx == current_zoom_object_id:
                        # cv2.putText(
                        #     annotated_frame, 
                        #     f"Tracking: {all_labels[idx]} ({prob:.0%})", 
                        #     (50, 50), 
                        #     cv2.FONT_HERSHEY_SIMPLEX, 
                        #     1, 
                        #     (0, 255, 0), 
                        #     2
                        # )
                        break
        else:
            annotated_frame = frame

        cv2.imshow('Object Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    objek_deteksi()