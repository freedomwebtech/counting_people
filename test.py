# Import required libraries
import cv2
from ultralytics import YOLO
import cvzone

# Load YOLOv8 model
model = YOLO('yolo12n.pt')

# Define vertical line's X position
line_x = 278

# Track previous center positions
track_history = {}

# IN/OUT counters
in_count = 0
out_count = 0

# Open video file or webcam
cap = cv2.VideoCapture("vid1.mp4")  # Use 0 for webcam

# Define the mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")
        
# Create a named OpenCV window and set the mouse callback
cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)
frame_count=0 
while True:
    # Read video frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 2 != 0:
        continue 
    frame = cv2.resize(frame, (1020, 600))
    

    # Detect and track persons (class 0)
    results = model.track(frame, persist=True, classes=[0])

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw box and center
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            if track_id in track_history:
                prev_cx, prev_cy = track_history[track_id]

                #Check if person moved from left to right across the line (IN)
                if (prev_cx < line_x <= cx) and abs(cy - prev_cy) < 50:
                    print(f"ID {track_id} crossed IN")
                    in_count += 1
                    cv2.putText(frame, "IN", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # Check if person moved from right to left across the line (OUT)    
                elif (prev_cx > line_x >= cx) and abs(cy - prev_cy) < 50:
                    print(f"ID {track_id} crossed OUT")
                    out_count += 1
                    cv2.putText(frame, "OUT", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Update tracking history
            track_history[track_id] = (cx, cy)

    # Draw the crossing line
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 2)
    print(frame.shape)

    # Display counts using cvzone's putTextRect
    cvzone.putTextRect(frame, f'IN: {in_count}', (40,60), scale=2, thickness=2, colorT=(255, 255, 255), colorR=(0, 128, 0))
    cvzone.putTextRect(frame, f'OUT: {out_count}', (40,100), scale=2, thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 255))

    # Show the frame
    cv2.imshow("RGB", frame)

    # Press ESC to exit
    if cv2.waitKey(0) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
