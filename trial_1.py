'''
DATE: 6/30/2025

LOGS

SUCCESS 
- CAN DETECT PERSON, OBJECT, CREATE BORDER
- IF PERSON INSIDE BORDER, STATUS = WARNING
- IF PERSON TOUCH OBJECT IN BORDER = DANGER

ISSUES
- CANNOT EXIT PROGRAM
- BUFFERING IS NOT SMOOTH

'''
import cv2
import torch
from ultralytics import YOLO

# Load YOLOv5n model (pretrained)
model = YOLO("yolov8n.pt")  # Ultralytics version; can replace with yolov5 if you use older one

# Define green border region (x1, y1, x2, y2)
border_top_left = (150, 100)
border_bottom_right = (500, 400)

# Define classes
PERSON_CLASS_ID = 0
OBJECT_CLASS_IDS = [39, 41, 67, 73]  # Examples: bottle, cup, etc.

# Start video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, verbose=False)[0]

    persons = []
    objects = []

    for box in result.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == PERSON_CLASS_ID:
            persons.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for person
        elif cls in OBJECT_CLASS_IDS:
            objects.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for object

    # Draw green border
    cv2.rectangle(frame, border_top_left, border_bottom_right, (0, 255, 0), 4)

    # Determine state
    state = "safe"
    person_in_border = False

    def intersects_border(x1, y1, x2, y2):
        bx1, by1 = border_top_left
        bx2, by2 = border_bottom_right
        return not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2)

    def boxes_intersect(box1, box2):
        xa1, ya1, xa2, ya2 = box1
        xb1, yb1, xb2, yb2 = box2
        return not (xa2 < xb1 or xa1 > xb2 or ya2 < yb1 or ya1 > yb2)

    for p_box in persons:
        if intersects_border(*p_box):
            person_in_border = True
            for o_box in objects:
                if boxes_intersect(p_box, o_box):
                    state = "danger"
                    break
            if state != "danger":
                state = "warning"

    # Show state on screen
    color = (0, 255, 0) if state == "safe" else (0, 255, 255) if state == "warning" else (0, 0, 255)
    cv2.putText(frame, f"Status: {state.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, color, 3)

    cv2.imshow("Border Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
