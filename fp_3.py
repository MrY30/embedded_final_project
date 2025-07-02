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

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)

    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
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

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

drawing = False
border_polygon = []
detection_active = False
object_buffer = deque(maxlen=10)
initial_object_count = None

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
        print("Draw mode: Click to define polygon border.")
        border_polygon = []
        drawing = True
        detection_active = False
        initial_object_count = None
    elif key == ord('s'):
        print("Detection started.")
        drawing = False
        detection_active = True
    elif key == ord('r'):
        print("Reset: Border cleared.")
        border_polygon = []
        drawing = False
        detection_active = False
        initial_object_count = None
    elif key == ord('q') or key == 27:
        print("Exiting.")
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
            if cls == PERSON_CLASS_ID:
                persons.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif cls in OBJECT_CLASS_IDS:
                current_objects.append((x1, y1, x2, y2))

        object_buffer.append(current_objects)
        all_objects = [box for buffer in object_buffer for box in buffer]

        def remove_duplicates(boxes, iou_threshold=0.3):
            final = []
            for box in boxes:
                if not any(boxes_intersect(box, b) for b in final):
                    final.append(box)
            return final

        objects = remove_duplicates(all_objects)

        for x1, y1, x2, y2 in objects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        state = "safe"
        for p_box in persons:
            if box_inside_polygon(*p_box, border_polygon):
                state = "warning"
                for o_box in objects:
                    if boxes_intersect(p_box, o_box):
                        GPIO.output(LED_PIN, GPIO.HIGH)
                        state = "danger"
                        break
                break

        objects_inside_border = 0
        for obj_box in objects:
            if box_inside_polygon(*obj_box, border_polygon):
                objects_inside_border += 1

        if initial_object_count is None:
            initial_object_count = objects_inside_border

        if objects_inside_border < initial_object_count:
            state = "stolen"
            GPIO.output(LED_PIN, GPIO.HIGH)

        if state == "safe":
            color = (0, 255, 0)
            GPIO.output(LED_PIN, GPIO.LOW)
        elif state == "warning":
            color = (0, 255, 255)
            GPIO.output(LED_PIN, GPIO.LOW)
        elif state == "danger":
            color = (0, 0, 255)
        elif state == "stolen":
            color = (0, 0, 128)

        cv2.putText(frame, f"Status: {state.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"Objects in border: {objects_inside_border}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (29, 56, 30), 1)

    cv2.putText(frame, "Press 'd' to draw, 's' to start, 'r' to reset, 'q' to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (39, 18, 52), 1)

    cv2.imshow("Border Monitor", frame)

print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
