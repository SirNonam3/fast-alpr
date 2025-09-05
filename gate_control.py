import cv2
from fast_alpr import ALPR

import relaylib

def trigger_gate():
    relay = relaylib.Relay(port="COM3")
    relay.on(1)   # close relay contact
    time.sleep(1)
    relay.off(1)  # open relay contact

# Plates allowed to open the gate
AUTHORIZED_PLATES = {"5AU5341", "34ABC123", "06XYZ789"}

# Initialize ALPR
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model"
)

# Connect to IP camera (replace with your RTSP/HTTP stream)
cap = cv2.VideoCapture("rtsp://admin:admin@192.168.1.35:554/stream")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 15 != 0:   # Only process every 5th frame (~6 FPS)
        continue

    results = alpr.predict(frame)

    for res in results:
        plate = res.ocr.text.strip()
        confidence = res.ocr.confidence

        print(f"Detected: {plate} (conf: {confidence:.2f})")

        # Check against authorized list
        if plate in AUTHORIZED_PLATES and confidence > 0.9:
            print(f"✅ Gate Opened for {plate}")
            trigger_gate()
            # ---- PLACE YOUR GATE CONTROL CODE HERE ----
            # e.g., send a signal to a relay, API call, etc.
            # trigger_gate()
    
    # Show live annotated video (optional)
    annotated = alpr.draw_predictions(frame)
    cv2.imshow("ALPR Gate System", annotated)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
