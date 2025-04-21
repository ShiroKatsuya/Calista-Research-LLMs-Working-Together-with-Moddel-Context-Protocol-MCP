import torch
import cv2
import numpy as np
import supervision as sv

import threading
import emoji
import time
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from colution import calculate_destruction_probability
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
            outputs, target_sizes=[(h, w)], threshold=0.1)

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
            enhanced_labels = []
            high_prob_objects = []
            
            for i, label in enumerate(labels):
                prob = calculate_destruction_probability(detections[i:i+1], frame.shape)
                enhanced_labels.append(f"{label}-T.Instant Destruction: {prob:.0%}")
                
                # Track all objects with their probability
                high_prob_objects.append((i, prob, detections[i]))
            
            # Sort all objects by probability (highest first)
            high_prob_objects.sort(key=lambda x: x[1], reverse=True)
            
            # Determine if we need to change zoom target
            current_target_valid = False
            
            # Check if current zoom target still has ≥80% probability
            if current_zoom_object_id is not None:
                for idx, prob, detection in high_prob_objects:
                    if idx == current_zoom_object_id and prob >= 0.7:
                        current_target_valid = True
                        zoom_target = detection
                        target_zoom_level = 2.0  # Full zoom
                        break
            
            # If current target is not valid (below 80% or not found)
            if not current_target_valid:
                # Find a new target with ≥80% probability
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
            
            # Apply zoom if needed
            display_frame = frame.copy()
            if current_zoom_level > 1.01 and zoom_target is not None:
                # Get the box coordinates for the target
                x1, y1, x2, y2 = zoom_target.xyxy[0]
                
                # Calculate center of the box - convert to integers
                center_x = int((x1 + x2) // 2)
                center_y = int((y1 + y2) // 2)
                
                # Calculate box dimensions
                box_width = int(x2 - x1)
                box_height = int(y2 - y1)
                
                # Calculate zoom area based on current_zoom_level
                zoom_factor = 1 + (current_zoom_level - 1) * 2  # Convert zoom level to area factor
                zoom_width = int(box_width * zoom_factor)
                zoom_height = int(box_height * zoom_factor)
                
                # Ensure the zoom area doesn't exceed frame dimensions
                x1_zoom = int(max(0, center_x - zoom_width // 2))
                y1_zoom = int(max(0, center_y - zoom_height // 2))
                x2_zoom = int(min(w, center_x + zoom_width // 2))
                y2_zoom = int(min(h, center_y + zoom_height // 2))
                
                # Extract and resize the region to create zoom effect
                zoom_region = frame[y1_zoom:y2_zoom, x1_zoom:x2_zoom]
                if zoom_region.size > 0:  # Make sure we have a valid region
                    display_frame = cv2.resize(zoom_region, (w, h))
                    
                    # Re-calculate annotation coordinates for zoomed view
                    zoom_detections = sv.Detections(
                        xyxy=detections.xyxy.copy(),
                        confidence=detections.confidence.copy() if detections.confidence is not None else None,
                        class_id=detections.class_id.copy() if detections.class_id is not None else None,
                        tracker_id=detections.tracker_id.copy() if detections.tracker_id is not None else None
                    )
                    
                    for i in range(len(zoom_detections.xyxy)):
                        # Transform coordinates to zoomed space
                        box = zoom_detections.xyxy[i]
                        # Scale coordinates to match the zoomed view
                        new_box = [
                            (box[0] - x1_zoom) * w / (x2_zoom - x1_zoom),
                            (box[1] - y1_zoom) * h / (y2_zoom - y1_zoom),
                            (box[2] - x1_zoom) * w / (x2_zoom - x1_zoom),
                            (box[3] - y1_zoom) * h / (y2_zoom - y1_zoom)
                        ]
                        zoom_detections.xyxy[i] = np.array(new_box)
                    
                    annotated_frame = sv.BoxAnnotator().annotate(display_frame, zoom_detections)
                    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, zoom_detections, labels=enhanced_labels)
                else:
                    # If zoom region is invalid, use normal view
                    annotated_frame = sv.BoxAnnotator().annotate(display_frame, detections)
                    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels=enhanced_labels)
            else:
                # Normal non-zoomed view
                annotated_frame = sv.BoxAnnotator().annotate(display_frame, detections)
                annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels=enhanced_labels)

            # Add info text about currently tracked object
            if current_zoom_object_id is not None and high_prob_objects:
                for idx, prob, _ in high_prob_objects:
                    if idx == current_zoom_object_id:
                        cv2.putText(
                            annotated_frame, 
                            f"Tracking: {labels[idx]} ({prob:.0%})", 
                            (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (0, 255, 0), 
                            2
                        )
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