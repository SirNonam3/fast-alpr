from fast_alpr import ALPR
import cv2

# Initialize ALPR with default models
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model"
)

# Load the test image (adjust path if needed)
image = cv2.imread("testfoto.jpg")

# Run prediction
results = alpr.predict(image)
print("Detected plates:", results)

# Draw predictions on the image
annotated = alpr.draw_predictions(image)
cv2.imwrite("output.png", annotated)
print("Results saved to output.png")
