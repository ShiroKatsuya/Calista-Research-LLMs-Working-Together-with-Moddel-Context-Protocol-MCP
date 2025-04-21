import numpy as np
def calculate_destruction_probability(detection, frame_shape):
    """Calculate probability of destroying target based on distance from center and object size"""
    # Get frame center
    center_x = frame_shape[1] / 2
    center_y = frame_shape[0] / 2
    
    # Get detection box center and dimensions
    box = detection.xyxy[0]
    box_center_x = (box[0] + box[2]) / 2
    box_center_y = (box[1] + box[3]) / 2
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    
    # Calculate Euclidean distance from center
    distance = np.sqrt((center_x - box_center_x)**2 + (center_y - box_center_y)**2)
    
    # Calculate diagonal distance of the frame (true maximum distance)
    max_distance = np.sqrt(frame_shape[1]**2 + frame_shape[0]**2) / 2
    normalized_distance = distance / max_distance
    
    # Calculate object size factor (larger objects are easier to target)
    object_area = box_width * box_height
    frame_area = frame_shape[1] * frame_shape[0]
    size_factor = min(0.5, object_area / frame_area * 10)  

    # Objects very close to center have high probability, rapidly decreasing as they move away
    sigma = 0.3  
    probability = np.exp(-(normalized_distance**2) / (2 * sigma**2))
    

    probability = min(0.99, probability + size_factor)
    
    return probability
