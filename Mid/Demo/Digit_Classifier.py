import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained CNN model
model = load_model('digit_cnn_mdl.h5')

def preprocess_image(image):
    """
    Preprocess webcam image to match MNIST format:
    - Grayscale
    - Adaptive threshold (white digit on black background)
    - Increase contrast
    - Resize to 28x28
    """
    # Convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Optional: blur to reduce noise
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Adaptive threshold for thin pen strokes
    img_thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Increase contrast for faint strokes
    img_contrast = cv2.convertScaleAbs(img_thresh, alpha=2.0, beta=0)

    # Optional: dilate to thicken strokes
    kernel = np.ones((2, 2), np.uint8)
    img_dilated = cv2.dilate(img_contrast, kernel, iterations=1)

    # Resize to 28x28 for CNN
    img_resized = cv2.resize(img_dilated, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize
    img_normalized = img_resized / 255.0
    img_normalized = img_normalized.reshape(1, 28, 28, 1)

    return img_normalized

def predict_digit(image, model):
    """
    Predict digit and probability from preprocessed image
    """
    img_preprocessed = preprocess_image(image)
    prediction = model.predict(img_preprocessed, verbose=0)
    prob = np.max(prediction)
    class_index = np.argmax(prediction, axis=1)[0]

    # Low confidence threshold
    if prob < 0.80:
        class_index = 0
        prob = 0

    return class_index, prob

# Open webcam
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

bbox_size = (40, 40)  # Cropped square size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_copy = frame.copy()
    # Define square bounding box at the center
    bbox = [
        (int(WIDTH//2 - bbox_size[0]//2), int(HEIGHT//2 - bbox_size[1]//2)),
        (int(WIDTH//2 + bbox_size[0]//2), int(HEIGHT//2 + bbox_size[1]//2))
    ]

    # Crop the digit region
    img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

    # Show enlarged cropped area
    cv2.imshow("Cropped Digit", cv2.resize(img_cropped, (200, 200)))

    # Predict digit
    result, probability = predict_digit(img_cropped, model)

    # Overlay prediction and probability
    cv2.putText(frame_copy, f"Prediction: {result}", (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_copy, f"Probability: {probability:.2f}", (40, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Draw bounding box
    cv2.rectangle(frame_copy, bbox[0], bbox[1], (0, 100, 100), 2)
    cv2.imshow("Digit Classifier", frame_copy)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
