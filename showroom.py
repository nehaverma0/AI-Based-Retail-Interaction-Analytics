import cv2
import numpy as np
from ultralytics import YOLO
import csv
import os
import math
import torch

# --- CONFIGURATION (You MUST change this section) ---

# 1. PATHS
VIDEO_PATH = r"F:\AI-Based Retail Interaction Analytics Project & Report File\Retail Analytics Project\showroom vdo.mp4"
MODEL_PATH = r"F:\AI-Based Retail Interaction Analytics Project & Report File\Retail Analytics Project\showroom pt.pt"
OUTPUT_VIDEO_PATH = "output_interactions_fast.avi"
CSV_LOG_PATH = "interaction_log_fast.csv"

# 2. CLASS IDs
CLASS_ID_CUSTOMER = 0
CLASS_ID_EMPLOYEE = 1
CLASS_MAP = {
    CLASS_ID_CUSTOMER: "Customer",
    CLASS_ID_EMPLOYEE: "Employee"
}
COLOR_MAP = {
    CLASS_ID_CUSTOMER: (255, 100, 0),
    CLASS_ID_EMPLOYEE: (0, 165, 255)
}

# 3. INTERACTION LOGIC
DISTANCE_THRESHOLD_PIXELS = 150
INTERACTION_TIME_THRESHOLD_S = 2.0

# 4. PERFORMANCE
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
frame_skip = 2         # process every 2nd frame (increase for more speed)
resize_width = 640     # lower resolution for faster inference
resize_height = 360

print(f"Using device: {DEVICE}")

# Load YOLO model
print("Loading model...")
model = YOLO(MODEL_PATH)
if DEVICE.startswith('cuda'):
    model.model.half()  # use half precision if GPU available
print("Model loaded.")

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Cannot open video file at {VIDEO_PATH}")
    exit()

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out_fps = fps // 2  # reduce output FPS to save processing time

out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'XVID'),
                      out_fps, (frame_width, frame_height))

# Setup CSV log
if not os.path.exists(CSV_LOG_PATH):
    with open(CSV_LOG_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['customer_id', 'employee_id', 'start_time_s', 'stop_time_s', 'duration_s'])

# State management
active_interactions = {}
total_unique_customers = set()

print("Processing video...")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Skip frames for speed
    if frame_count % frame_skip != 0:
        continue

    video_time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # Resize for faster inference
    frame_resized = cv2.resize(frame, (resize_width, resize_height))

    # YOLO tracking
    results = model.track(frame_resized, persist=True, device=DEVICE, verbose=False)

    if results[0].boxes is None or results[0].boxes.id is None:
        out.write(frame)
        display_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))
        cv2.imshow("Retail Analytics", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Extract detection info
    boxes = results[0].boxes.xyxy.cpu().numpy()
    tracking_ids = results[0].boxes.id.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    customers_in_frame = []
    employees_in_frame = []

    # Map detections to customers/employees
    for box, track_id, cls_id in zip(boxes, tracking_ids, class_ids):
        if cls_id not in CLASS_MAP:
            continue
        x1, y1, x2, y2 = box
        person_point = (int((x1 + x2) / 2 * (frame_width / resize_width)),
                        int(y2 * (frame_height / resize_height)))
        person_data = {'id': int(track_id), 'point': person_point, 'box': (int(x1 * (frame_width / resize_width)),
                                                                          int(y1 * (frame_height / resize_height)),
                                                                          int(x2 * (frame_width / resize_width)),
                                                                          int(y2 * (frame_height / resize_height)))}
        if cls_id == CLASS_ID_CUSTOMER:
            customers_in_frame.append(person_data)
            total_unique_customers.add(person_data['id'])
        elif cls_id == CLASS_ID_EMPLOYEE:
            employees_in_frame.append(person_data)

    # --- Interaction Logic ---
    current_frame_interactions = set()

    for cust in customers_in_frame:
        for emp in employees_in_frame:
            dist = math.dist(cust['point'], emp['point'])
            if dist < DISTANCE_THRESHOLD_PIXELS:
                interaction_key = (cust['id'], emp['id'])
                current_frame_interactions.add(interaction_key)

                if interaction_key not in active_interactions:
                    active_interactions[interaction_key] = {'start_time': video_time_s, 'confirmed': False}
                else:
                    timer = active_interactions[interaction_key]
                    elapsed = video_time_s - timer['start_time']
                    if elapsed > INTERACTION_TIME_THRESHOLD_S:
                        timer['confirmed'] = True
                        cv2.line(frame, cust['point'], emp['point'], (0, 255, 0), 2)

    # Clean up ended interactions
    ended_interactions = []
    for interaction_key, timer in active_interactions.items():
        if interaction_key not in current_frame_interactions:
            if timer['confirmed']:
                cust_id, emp_id = interaction_key
                start_time = timer['start_time']
                stop_time = video_time_s
                duration = stop_time - start_time
                with open(CSV_LOG_PATH, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([cust_id, emp_id, f"{start_time:.2f}", f"{stop_time:.2f}", f"{duration:.2f}"])
            ended_interactions.append(interaction_key)

    for key in ended_interactions:
        del active_interactions[key]

    # --- Drawing on Frame ---
    all_people = customers_in_frame + employees_in_frame
    for person in all_people:
        x1, y1, x2, y2 = person['box']
        person_id = person['id']
        cls_id = CLASS_ID_CUSTOMER if person in customers_in_frame else CLASS_ID_EMPLOYEE
        label = CLASS_MAP[cls_id]
        color = COLOR_MAP[cls_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label}: {person_id}"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Dashboard
    cv2.putText(frame, f"Total Unique Customers: {len(total_unique_customers)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Active Interactions: {len(active_interactions)}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out.write(frame)
    display_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))
    cv2.imshow("Retail Analytics (Fast Mode)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Processing complete.")
print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")
print(f"Interaction log saved to: {CSV_LOG_PATH}")
