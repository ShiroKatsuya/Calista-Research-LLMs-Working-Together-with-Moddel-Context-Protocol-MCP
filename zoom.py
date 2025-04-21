import cv2
import numpy as np
import supervision as sv

def apply_zoom(frame, zoom_target, current_zoom_level, detections, enhanced_labels, multi_object_mode=False):
    h, w = frame.shape[:2]
    display_frame = frame.copy()
    
    if current_zoom_level > 1.01:
        if multi_object_mode and zoom_target and isinstance(zoom_target, list) and len(zoom_target) > 0:
            # Find the bounding box that contains all high probability objects
            min_x = min_y = float('inf')
            max_x = max_y = 0
            
            for target in zoom_target:
                x1, y1, x2, y2 = target.xyxy[0]
                min_x = min(min_x, x1)
                min_y = min(min_y, y1)
                max_x = max(max_x, x2)
                max_y = max(max_y, y2)
            
            # Calculate center of the combined area
            center_x = int((min_x + max_x) // 2)
            center_y = int((min_y + max_y) // 2)
            
            # Calculate dimensions of the combined area with some padding
            combined_width = int((max_x - min_x) * 1.2)  # Add 20% padding
            combined_height = int((max_y - min_y) * 1.2)  # Add 20% padding
            
            # Calculate zoom area based on current_zoom_level (less zoom than single object mode)
            zoom_factor = current_zoom_level  # More moderate zoom for multiple objects
            zoom_width = int(combined_width * zoom_factor)
            zoom_height = int(combined_height * zoom_factor)
            
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
        
        elif zoom_target is not None:  # Single object zoom (original behavior)
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
            # Normal non-zoomed view when we have no valid target
            annotated_frame = sv.BoxAnnotator().annotate(display_frame, detections)
            annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels=enhanced_labels)
    else:
        # Normal non-zoomed view
        annotated_frame = sv.BoxAnnotator().annotate(display_frame, detections)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels=enhanced_labels)
    
    return annotated_frame