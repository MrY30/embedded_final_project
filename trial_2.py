'''
DATE: 6/30/2025

LOGS

SUCCESS 
- CAN DETECT PERSON, OBJECT, CREATE BORDER
- IF PERSON INSIDE BORDER, STATUS = WARNING
- IF PERSON TOUCH OBJECT IN BORDER = DANGER
- EXITS PROGRAM
- BUFFERING IS SMOOTH

NEXT STEP:
- CUSTOMIZABLE BORDER

'''

import cv2
from ultralytics import YOLO
from collections import deque

# Load YOLO model
model = YOLO("yolov8n.pt")

# Border box
border_top_left = (150, 100)
border_bottom_right = (500, 400)

# Classes
PERSON_CLASS_ID = 0
OBJECT_CLASS_IDS = [39, 41, 67, 73]  # Example: bottle, cup, etc.

# Object smoothing (keep last 10 frames' detections)
object_buffer = deque(maxlen=10)

# Start camera
cap = cv2.VideoCapture(0)

def intersects_border(x1, y1, x2, y2):
    bx1, by1 = border_top_left
    bx2, by2 = border_bottom_right
    return not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2)

def boxes_intersect(box1, box2):
    xa1, ya1, xa2, ya2 = box1
    xb1, yb1, xb2, yb2 = box2
    return not (xa2 < xb1 or xa1 > xb2 or ya2 < yb1 or ya1 > yb2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, verbose=False)[0]

    persons = []
    current_objects = []

    # Parse detection results
    for box in result.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == PERSON_CLASS_ID:
            persons.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        elif cls in OBJECT_CLASS_IDS:
            current_objects.append((x1, y1, x2, y2))

    # Add to object buffer and average across frames
    object_buffer.append(current_objects)
    all_objects = [box for buffer in object_buffer for box in buffer]

    # Filter duplicates (optional improvement)
    def remove_duplicates(boxes, iou_threshold=0.3):
        final = []
        for box in boxes:
            if not any(boxes_intersect(box, b) for b in final):
                final.append(box)
        return final

    objects = remove_duplicates(all_objects)

    # Draw object boxes
    for x1, y1, x2, y2 in objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw green border
    cv2.rectangle(frame, border_top_left, border_bottom_right, (0, 255, 0), 4)

    # Detect status
    state = "safe"

    for p_box in persons:
        if intersects_border(*p_box):
            for o_box in objects:
                if boxes_intersect(p_box, o_box):
                    state = "danger"
                    break
            if state != "danger":
                state = "warning"

    # Display status
    color = (0, 255, 0) if state == "safe" else (0, 255, 255) if state == "warning" else (0, 0, 255)
    cv2.putText(frame, f"Status: {state.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Border Monitor", frame)

    key = cv2.waitKey(10)
    if key == ord('q') or key == 27:  # 'q' or ESC to quit
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
