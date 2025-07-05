import os 
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# import RPi.GPIO as GPIO

# GPIO.setmode(GPIO.BCM)
# ALARM_PIN = 14
# GPIO.setup(ALARM_PIN, GPIO.OUT)
# GPIO.output(ALARM_PIN, GPIO.LOW)

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model (e.g. "runs/detect/train/weights/best.pt")')
parser.add_argument('--source', required=True, help='Image/video source (e.g. "usb0", "test.jpg", or "testvid.mp4")')
parser.add_argument('--thresh', default=0.5, help='Minimum confidence threshold')
parser.add_argument('--resolution', default=None, help='Resolution WxH (e.g. "640x480")')
parser.add_argument('--record', action='store_true', help='Save video to "demo1.avi"')
args = parser.parse_args()

model_path = args.model
img_source = args.source
user_res = args.resolution
record = args.record

# --- Model Load ---
if not os.path.exists(model_path):
    print('ERROR: Model not found.')
    sys.exit(0)
model = YOLO(model_path, task='detect')

# --- Source Handling ---
img_ext_list = ['.jpg','.jpeg','.png','.bmp']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    source_type = 'image' if ext in img_ext_list else 'video' if ext in vid_ext_list else sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Invalid input: {img_source}')
    sys.exit(0)

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
if source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# --- State Variables ---
drawing = False
border_polygon = []
detection_active = False
object_buffer = deque(maxlen=10)
initial_object_classes = None

PERSON_CLASS_ID = 0
OBJECT_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                    61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]

def remove_duplicates(boxes):
    final = []
    for box in boxes:
        found = False
        for b in final:
            if boxes_intersect(box[:4], b[:4]):
                found = True
                break
        if not found:
            final.append(box)
    return final

def draw_border(event, x, y, flags, param):
    global border_polygon, drawing
    if detection_active:
        return
    if event == cv2.EVENT_LBUTTONDOWN and drawing:
        border_polygon.append((x, y))

def box_inside_polygon(x1, y1, x2, y2, polygon, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    poly = np.array([polygon], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)
    
    person_mask = np.zeros_like(mask)
    cv2.rectangle(person_mask, (x1, y1), (x2, y2), 255, -1)
    
    intersection = cv2.bitwise_and(mask, person_mask)
    
    return np.any(intersection)

def boxes_intersect(box1, box2):
    xa1, ya1, xa2, ya2 = box1
    xb1, yb1, xb2, yb2 = box2
    return not (xa2 < xb1 or xa1 > xb2 or ya2 < yb1 or ya1 > yb2)

cv2.namedWindow("Border Monitor")
cv2.setMouseCallback("Border Monitor", draw_border)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        border_polygon = []
        drawing = True
        detection_active = False
        initial_object_classes = None
    elif key == ord('s'):
        drawing = False
        detection_active = True
    elif key == ord('r'):
        border_polygon = []
        drawing = False
        detection_active = False
        initial_object_classes = None
    elif key == ord('q') or key == 27:
        break

    for pt in border_polygon:
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)
    if len(border_polygon) >= 2:
        for i in range(len(border_polygon)):
            pt1 = border_polygon[i]
            pt2 = border_polygon[(i + 1) % len(border_polygon)]
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    
    persons = []
    objects = []
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
        objects = all_objects
        
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
            if box_inside_polygon(x1, y1, x2, y2, border_polygon, frame.shape):
                objects_inside.append(obj)

        # State machine for danger/warning/safe
        state = "safe"
        person_inside = False
        danger_triggered = False
        
        for p_box in persons:
            if box_inside_polygon(*p_box, border_polygon, frame.shape):
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
            #GPIO.output(ALARM_PIN, GPIO.LOW)
        elif person_inside:
            state = "warning"
            #GPIO.output(ALARM_PIN, GPIO.LOW)
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
                #GPIO.output(ALARM_PIN, GPIO.HIGH)
        # === END NEW LOGIC ===

        color_map = {
            "safe": (0, 255, 0),
            "warning": (0, 255, 255),
            "danger": (0, 0, 255),
            "stolen": (0, 0, 128)
        }

        cv2.putText(frame, f"Status: {state.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[state], 1)

    cv2.putText(frame, "Press 'd' to draw, 's' to start, 'r' to reset, 'q' to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
    cv2.imshow("Border Monitor", frame)

# --- Cleanup ---
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()



# python final_code.py --model=yolo11n_ncnn_model --source=usb0 --resolution=1280x720
