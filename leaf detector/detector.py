import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("plantVillage_model_10.h5")

# Load labels
with open("plantvillage_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green color range (adjust if needed for your lighting/leaf)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Mask only green parts
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours of green regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Take the largest green area (assume it's the leaf)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        if w * h > 5000:  # Ignore very small green spots
            leaf_roi = frame[y:y+h, x:x+w]

            # Preprocess for model
            img = cv2.resize(leaf_roi, (128, 128))
            img = np.expand_dims(img, axis=0) / 255.0

            # Predict
            predictions = model.predict(img)
            class_index = np.argmax(predictions[0])
            confidence = predictions[0][class_index]

            # ✅ Map all labels to only Healthy/Diseased
            if "healthy" in labels[class_index].lower():
                final_label = "Healthy"
                color = (0, 255, 0)  # Green
            else:
                final_label = "Diseased"
                color = (0, 0, 255)  # Red

            # Draw box + label on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{final_label} ({confidence*100:.1f}%)",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)
        else:
            cv2.putText(frame, "Searching...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "Searching...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show output
    cv2.imshow("Leaf Detection + Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()