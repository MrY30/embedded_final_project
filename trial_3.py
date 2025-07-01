'''
DATE: 7/1/2025

LOGS

SUCCESS 
- CAN DETECT PERSON, OBJECT, CREATE BORDER
- IF PERSON INSIDE BORDER, STATUS = WARNING
- IF PERSON TOUCH OBJECT IN BORDER = DANGER
- EXITS PROGRAM
- BUFFERING IS SMOOTH
- BORDER IS CUSTOMIZABLE

NEXT STEP:
- ARDUINO ALARM SYSTEM

'''

import cv2
from ultralytics import YOLO
from collections import deque

# Load YOLO model
model = YOLO("yolov8n.pt")

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

# Start camera
cap = cv2.VideoCapture(0)
cv2.namedWindow("Border Monitor")
cv2.setMouseCallback("Border Monitor", draw_border)

import numpy as np

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

        color = (0, 255, 0) if state == "safe" else (0, 255, 255) if state == "warning" else (0, 0, 255)
        cv2.putText(frame, f"Status: {state.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Instructions overlay
    cv2.putText(frame, "Press 'd' to draw, 's' to start, 'r' to reset, 'q' to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Border Monitor", frame)

cap.release()
cv2.destroyAllWindows()