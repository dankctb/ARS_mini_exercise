import cv2
import numpy as np

def check_object_frame(frame):
    h = frame.shape[0]
    w = frame.shape[1]
    gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blurred_image = cv2.medianBlur(binary_image,5,0)
    center_area = blurred_image[int(h/5 - h/16) : int(h/5 + h/16), int(w/2 - w/5) : int(w/2 + w/5)]
    lower_area = blurred_image[int(h-h/16):, int(w/2 - w/6) : int(w/2 + w/6)]

    return int((np.sum(center_area) > 0)&(np.sum(lower_area) == 0 ))

def draw(table_data, detection):

    image_width = 400
    image_height = 500
    image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255


    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75  
    font_thickness = 1  
    text_color = (0, 0, 0)  # Black 

    # Add the first table title
    table_title = "Counter"
    text_size = cv2.getTextSize(table_title, font, font_scale, font_thickness)[0]
    title_position = ((image_width - text_size[0]) // 2, 50)
    cv2.putText(image, table_title, title_position, font, font_scale, text_color, font_thickness)

    # Adjust cell height for smaller image size
    cell_height = 30

    # Add the first table
    table_start = (50, title_position[1] + text_size[1] + 10)
    
    for i, (name, value) in enumerate(table_data):
        cell_position = (table_start[0], table_start[1] + i * cell_height)
        cv2.putText(image, f"{name}: {value}", cell_position, font, font_scale, text_color, font_thickness)

    # Add the second table title
    table_title2 = "Detection"
    text_size2 = cv2.getTextSize(table_title2, font, font_scale, font_thickness)[0]
    title_position2 = ((image_width - text_size2[0]) // 2, table_start[1] + len(table_data) * cell_height + 20)
    cv2.putText(image, table_title2, title_position2, font, font_scale, text_color, font_thickness)

    # Add the second table
    table_start2 = (50, title_position2[1] + text_size2[1] + 10)
    
    for i, (prev_obj, current_obj) in enumerate(detection):
        cell_position2 = (table_start2[0], table_start2[1] + i * cell_height)
        cv2.putText(image, f"{prev_obj}: {current_obj}", cell_position2, font, font_scale, text_color, font_thickness)
    
    return image

def create_bounding_box(frame, pre_frame):

    subtracted = cv2.subtract(frame, pre_frame)

    gray = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)
    
    blurred_image = cv2.medianBlur(gray,9,0)
        
    min_threshold = 20
    max_threshold = 80
    edges_subtract = cv2.Canny(blurred_image, min_threshold, max_threshold)

    points = list(np.where(edges_subtract > 0))
    
    cv2.rectangle(frame, (min(points[1]), min(points[0])), (max(points[1]), max(points[0])), (0, 0, 255), 3)

    return frame