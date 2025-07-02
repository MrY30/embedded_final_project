import cv2
import numpy as np
import onnxruntime as ort
from threading import Thread
from collections import deque

# --- Configuration ---
ONNX_MODEL_PATH = "yolov8n.onnx"  # Path to your converted ONNX model
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
CONF_THRESHOLD = 0.45 # Confidence threshold
IOU_THRESHOLD = 0.4 # Intersection over Union for non-maximum suppression

# --- Globals ---
border_polygon = []
detection_active = False
drawing = False
last_known_state = "safe"
last_known_persons = []
last_known_objects = []
last_objects_inside_border = 0

PERSON_CLASS_ID = 0
# List of common object class IDs from COCO dataset
OBJECT_CLASS_IDS = [24, 25, 26, 27, 28, 39, 40, 41, 42, 43,
                    44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                    54, 55, 56, 63, 64, 65, 66, 67, 73, 74,
                    75, 76, 77, 78, 79]

# --- Threaded Camera Capture ---
class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# --- ONNX Model Loading ---
try:
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    model_inputs = session.get_inputs()
    input_shape = model_inputs[0].shape
    input_name = model_inputs[0].name
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    print("Please ensure you have converted the YOLOv8 model to ONNX format.")
    exit()

# --- Utility Functions ---
def draw_border(event, x, y, flags, param):
    global border_polygon, drawing
    if detection_active: return
    if event == cv2.EVENT_LBUTTONDOWN and drawing:
        border_polygon.append((x, y))

def box_inside_polygon(box_center, polygon):
    poly_np = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly_np, box_center, False) >= 0

def boxes_intersect(box1, box2):
    xa1, ya1, xa2, ya2 = box1
    xb1, yb1, xb2, yb2 = box2
    return not (xa2 < xb1 or xa1 > xb2 or ya2 < yb1 or ya1 > yb2)

# --- Main Application ---
vs = WebcamStream(src=0).start()
cv2.namedWindow("Border Monitor")
cv2.setMouseCallback("Border Monitor", draw_border)

frame_count = 0

while True:
    frame = vs.read()
    if frame is None:
        continue
    
    display_frame = frame.copy()
    frame_height, frame_width, _ = display_frame.shape
    # Calculate scaling factors
    x_factor = frame_width / INPUT_WIDTH
    y_factor = frame_height / INPUT_HEIGHT

    key = cv2.waitKey(1) & 0xFF

    # --- Key Bindings ---
    if key == ord('d'):
        print("Draw mode: Click to define polygon border.")
        border_polygon = []; drawing = True; detection_active = False
    elif key == ord('s'):
        print("Detection started.")
        drawing = False; detection_active = True
    elif key == ord('r'):
        print("Reset: Border cleared.")
        border_polygon = []; drawing = False; detection_active = False
    elif key == ord('q') or key == 27:
        print("Exiting.")
        break

    # --- Detection Logic (runs every 3 frames) ---
    if detection_active and len(border_polygon) >= 3 and frame_count % 3 == 0:
        # Prepare frame for inference
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (INPUT_WIDTH, INPUT_HEIGHT))
        input_img = input_img.transpose(2, 0, 1) # HWC to CHW
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32) / 255.0

        # Run Inference
        outputs = session.run(None, {input_name: input_tensor})
        
        boxes, scores, class_ids = [], [], []
        # Process output from the ONNX model
        output_data = outputs[0][0].T # Transpose to get detections as rows
        for detection in output_data:
            class_probabilities = detection[4:]
            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id]
            
            if confidence > CONF_THRESHOLD:
                cx, cy, w, h = detection[:4]
                x1 = int((cx - w / 2) * x_factor)
                y1 = int((cy - h / 2) * y_factor)
                x2 = int((cx + w / 2) * x_factor)
                y2 = int((cy + h / 2) * y_factor)
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(confidence))
                class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
        
        persons, objects = [], []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                if class_ids[i] == PERSON_CLASS_ID:
                    persons.append(box)
                elif class_ids[i] in OBJECT_CLASS_IDS:
                    objects.append(box)
        
        # Determine Status
        state = "safe"
        for p_box in persons:
            p_center = ((p_box[0] + p_box[2]) // 2, (p_box[1] + p_box[3]) // 2)
            if box_inside_polygon(p_center, border_polygon):
                state = "warning"
                for o_box in objects:
                    if boxes_intersect(p_box, o_box):
                        state = "danger"
                        break
        
        objects_inside_border = sum(1 for o_box in objects if box_inside_polygon(((o_box[0] + o_box[2]) // 2, (o_box[1] + o_box[3]) // 2), border_polygon))
        
        # Update last known states
        last_known_persons, last_known_objects = persons, objects
        last_known_state = state
        last_objects_inside_border = objects_inside_border

    # --- Drawing and Display (runs every frame) ---
    if len(border_polygon) >= 2:
        pts = np.array(border_polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(display_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    for x1, y1, x2, y2 in last_known_persons:
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for x1, y1, x2, y2 in last_known_objects:
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    color = (0, 255, 0) if last_known_state == "safe" else (0, 255, 255) if last_known_state == "warning" else (0, 0, 255)
    cv2.putText(display_frame, f"Status: {last_known_state.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(display_frame, f"Objects in border: {last_objects_inside_border}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(display_frame, "d: Draw | s: Start | r: Reset | q: Quit", (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Border Monitor", display_frame)
    frame_count += 1

vs.stop()
cv2.destroyAllWindows()