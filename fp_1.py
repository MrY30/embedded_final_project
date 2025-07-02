import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

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
    
    # Set up recording
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

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()



# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# UPDATE ADD CODE FROM PROJECT
# Globals
drawing = False
border_polygon = []
detection_active = False
object_buffer = deque(maxlen=10)

PERSON_CLASS_ID = 0
OBJECT_CLASS_IDS = [24, 25, 26, 27, 28, 39, 40, 41, 42, 43, 
                    44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 
                    54, 55, 56, 63, 64, 65, 66, 67, 73, 74,
                    75, 76, 77, 78, 79]

# Mouse callback for drawing polygon
def draw_border(event, x, y, flags, param):
    global border_polygon, drawing
    if detection_active:
        return
    if event == cv2.EVENT_LBUTTONDOWN and drawing:
        border_polygon.append((x, y))

# Utility to check if box center is inside polygon
def box_inside_polygon(x1, y1, x2, y2, polygon):
    poly = np.array(polygon, dtype=np.int32)
    # Check each corner of the box
    corners = [
        (x1, y1),
        (x1, y2),
        (x2, y1),
        (x2, y2)
    ]
    for corner in corners:
        if cv2.pointPolygonTest(poly, corner, False) >= 0:
            return True
    return False

# Check if two boxes intersect
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

    # Key bindings
    if key == ord('d'):
        print("Draw mode: Click to define polygon border.")
        border_polygon = []
        drawing = True
        detection_active = False
    elif key == ord('s'):
        print("Detection started.")
        drawing = False
        detection_active = True
    elif key == ord('r'):
        print("Reset: Border cleared.")
        border_polygon = []
        drawing = False
        detection_active = False
    elif key == ord('q') or key == 27:
        print("Exiting.")
        break

    # Draw clicked points
    for pt in border_polygon:
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)  # small green dot

    # Draw polygon lines (if >= 2 points)
    if len(border_polygon) >= 2:
        for i in range(len(border_polygon)):
            pt1 = border_polygon[i]
            pt2 = border_polygon[(i + 1) % len(border_polygon)]
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)


    # Detection
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

        # Remove duplicates
        def remove_duplicates(boxes, iou_threshold=0.3):
            final = []
            for box in boxes:
                if not any(boxes_intersect(box, b) for b in final):
                    final.append(box)
            return final

        objects = remove_duplicates(all_objects)

        for x1, y1, x2, y2 in objects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Determine status
        state = "safe"

        for p_box in persons:
            if box_inside_polygon(*p_box, border_polygon):
                for o_box in objects:
                    if boxes_intersect(p_box, o_box):
                        state = "danger"
                        break
                if state != "danger":
                    state = "warning"
        
        # Count how many objects are inside the polygon
        objects_inside_border = 0
        for obj_box in objects:
            if box_inside_polygon(*obj_box, border_polygon):
                objects_inside_border += 1

        # Display Status
        color = (0, 255, 0) if state == "safe" else (0, 255, 255) if state == "warning" else (0, 0, 255)
        cv2.putText(frame, f"Status: {state.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Object counts
        cv2.putText(frame, f"Objects in border: {objects_inside_border}", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (29, 56, 30), 1)

    # Instructions overlay
    cv2.putText(frame, "Press 'd' to draw, 's' to start, 'r' to reset, 'q' to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (39, 18, 52), 1)

    cv2.imshow("Border Monitor", frame)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()
