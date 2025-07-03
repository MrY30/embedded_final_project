import os 
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
LED_PIN = 14
GPIO.setup(LED_PIN, GPIO.OUT)

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
OBJECT_CLASS_IDS = list(range(1, 80))

def draw_border(event, x, y, flags, param):
    global border_polygon, drawing
    if detection_active:
        return
    if event == cv2.EVENT_LBUTTONDOWN and drawing:
        border_polygon.append((x, y))

def box_inside_polygon(x1, y1, x2, y2, polygon):
    poly = np.array(polygon, dtype=np.int32)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return cv2.pointPolygonTest(poly, (center_x, center_y), False) >= 0

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

        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (255, 0, 0) if cls == PERSON_CLASS_ID else (0, 0, 255)
            label = f"{int(cls)}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if cls == PERSON_CLASS_ID:
                persons.append((x1, y1, x2, y2))
            elif cls in OBJECT_CLASS_IDS:
                if box_inside_polygon(x1, y1, x2, y2, border_polygon):
                    current_objects.append((x1, y1, x2, y2))

        object_buffer.append(current_objects)
        all_objects = [box for buffer in object_buffer for box in buffer]

        def remove_duplicates(boxes):
            final = []
            for box in boxes:
                if not any(boxes_intersect(box, b) for b in final):
                    final.append(box)
            return final

        objects = remove_duplicates(all_objects)

        state = "safe"
        person_inside = False
        danger_triggered = False

        for p_box in persons:
            if box_inside_polygon(*p_box, border_polygon):
                person_inside = True
                for o_box in objects:
                    if boxes_intersect(p_box, o_box):
                        danger_triggered = True
                        break
            if danger_triggered:
                break

        if danger_triggered:
            state = "danger"
            GPIO.output(LED_PIN, GPIO.HIGH)
        elif person_inside:
            state = "warning"
            GPIO.output(LED_PIN, GPIO.LOW)
        else:
            GPIO.output(LED_PIN, GPIO.LOW)

        if initial_object_classes is None:
            initial_object_classes = []
            for box in result.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if box_inside_polygon(x1, y1, x2, y2, border_polygon):
                    initial_object_classes.append(cls)

        current_object_classes = []
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if box_inside_polygon(x1, y1, x2, y2, border_polygon):
                current_object_classes.append(cls)

        if sorted(current_object_classes) != sorted(initial_object_classes):
            state = "stolen"
            GPIO.output(LED_PIN, GPIO.HIGH)

        color_map = {
            "safe": (0, 255, 0),
            "warning": (0, 255, 255),
            "danger": (0, 0, 255),
            "stolen": (0, 0, 128)
        }

        cv2.putText(frame, f"Status: {state.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[state], 1)

    cv2.putText(frame, "Press 'd' to draw, 's' to start, 'r' to reset, 'q' to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (39, 18, 52), 1)
    cv2.imshow("Border Monitor", frame)

if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()

#RUN THE CODE
# python fp_6.py --model=yolo11n_ncnn_model --source=usb0 --resolution=1280x720