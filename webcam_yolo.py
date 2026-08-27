
import cv2, torch
from ultralytics import YOLO

dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("no camera. check System Settings > Privacy & Security > Camera")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = model(frame, device=dev, imgsz=480, verbose=False)[0]
    cv2.imshow("yolo - press q to quit", r.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release(); cv2.destroyAllWindows()
