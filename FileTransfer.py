import os 
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ... (GPIO and argument parsing remains unchanged) ...

# --- Resolution and Recorder ---
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))
if record:
    if source_type not in ['video', 'usb'] or not user_res:
        print('Recording requires --resolution and video/camera source')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# --- Source Loader ---
# ... (source loading remains unchanged) ...

# --- State Variables ---
drawing = False
border_polygon = []
detection_active = False
object_buffer = deque(maxlen=10)
initial_object_classes = None

PERSON_CLASS_ID = 0
OBJECT_CLASS_IDS = list(range(1, 80))

# --- Modified remove_duplicates for 5-tuples ---
def remove_duplicates(boxes):
    final = []
    for box in boxes:
        found = False
        for b in final:
            # Only check first 4 elements (coordinates) for intersection
            if boxes_intersect(box[:4], b[:4]):
                found = True
                break
        if not found:
            final.append(box)
    return final

# ... (draw_border and helper functions remain unchanged) ...

cv2.namedWindow("Border Monitor")
cv2.setMouseCallback("Border Monitor", draw_border)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ... (keyboard handling remains unchanged) ...

    # Draw border points/lines
    for pt in border_polygon:
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)
    if len(border_polygon) >= 2:
        for i in range(len(border_polygon)):
            pt1 = border_polygon[i]
            pt2 = border_polygon[(i + 1) % len(border_polygon)]
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    persons = []
    if detection_active and len(border_polygon) >= 3:
        result = model(frame, verbose=False)[0]
        current_objects = []

        # Process detections - separate persons and objects
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls == PERSON_CLASS_ID:
                persons.append((x1, y1, x2, y2))
                # Draw person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "person", (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            elif cls in OBJECT_CLASS_IDS:
                # Store as 5-tuple: (x1, y1, x2, y2, cls)
                current_objects.append((x1, y1, x2, y2, cls))

        # Update object buffer with current frame's objects
        object_buffer.append(current_objects)
        # Flatten buffer and remove duplicates
        all_objects = [box for buffer in object_buffer for box in buffer]
        objects = remove_duplicates(all_objects)
        
        # Draw buffered objects
        for obj in objects:
            x1, y1, x2, y2, cls = obj
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{cls}", (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Create list of objects inside border (from buffer)
        objects_inside = []
        for obj in objects:
            x1, y1, x2, y2, cls = obj
            if box_inside_polygon(x1, y1, x2, y2, border_polygon):
                objects_inside.append(obj)

        # State machine for danger/warning/safe
        state = "safe"
        person_inside = False
        danger_triggered = False
        
        for p_box in persons:
            if box_inside_polygon(*p_box, border_polygon):
                person_inside = True
                # Check intersection with objects INSIDE border
                for o_box in objects_inside:
                    if boxes_intersect(p_box, o_box[:4]):
                        danger_triggered = True
                        break
            if danger_triggered:
                break

        if danger_triggered:
            state = "danger"
            # GPIO.output(LED_PIN, GPIO.HIGH)
        elif person_inside:
            state = "warning"
            # GPIO.output(LED_PIN, GPIO.LOW)
        # else: safe state remains

        # === NEW STOLEN DETECTION LOGIC ===
        # 1. Set initial baseline when first entering safe state
        if initial_object_classes is None and state == "safe":
            initial_object_classes = []
            for obj in objects_inside:
                _, _, _, _, cls = obj
                initial_object_classes.append(cls)
        
        # 2. Check for stolen objects only in safe state
        if state == "safe" and initial_object_classes is not None:
            # Get current object classes inside border
            current_object_classes = [obj[4] for obj in objects_inside]
            
            # Compare with initial baseline
            stolen_detected = False
            unique_classes = set(initial_object_classes)
            for cls in unique_classes:
                count_initial = initial_object_classes.count(cls)
                count_current = current_object_classes.count(cls)
                if count_current < count_initial:
                    stolen_detected = True
                    break
                    
            if stolen_detected:
                state = "stolen"
        # === END NEW LOGIC ===

        # Status display
        color_map = {
            "safe": (0, 255, 0),
            "warning": (0, 255, 255),
            "danger": (0, 0, 255),
            "stolen": (0, 0, 128)
        }
        cv2.putText(frame, f"Status: {state.upper()}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[state], 1)

    # ... (UI text and display remains unchanged) ...

# ... (cleanup remains unchanged) ...