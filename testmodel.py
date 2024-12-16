import os
import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import sys

# ==============================
# CONFIGURATION
# ==============================
CALIBRATION_TIME = 5    # seconds to collect each direction
BUFFER_TIME = 2         # seconds buffer before switching to next direction
LABELS = ["CENTER", "UP", "LEFT", "DOWN", "RIGHT"]
EPOCHS = 20             # more epochs for better training

# Colors in BGR format
COLORS = {
    "CENTER": (255, 0, 255), # Purple
    "UP": (0, 0, 255),       # Red
    "LEFT": (0, 255, 255),   # Yellow
    "DOWN": (0, 255, 0),     # Green
    "RIGHT": (255, 0, 0),    # Blue
    "BUFFER": (0, 0, 0)      # Black
}

# Define the landmarks to be used
RIGHT_EYE_REGION = [33, 133, 160, 158, 159, 144, 145, 153, 246]
LEFT_EYE_REGION = [263, 362, 387, 385, 380, 373, 374, 381, 382]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474, 475, 476, 477]
NOSE_LANDMARKS = [1, 4]

SELECTED_LANDMARKS = RIGHT_EYE_REGION + LEFT_EYE_REGION + RIGHT_IRIS + LEFT_IRIS + NOSE_LANDMARKS
NUM_CLASSES = len(LABELS)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==============================
# FUNCTIONS
# ==============================

def normalize_landmarks(landmarks, image_shape):
    """
    Extract selected landmarks, then normalize:
    1. Translate so that nose landmark #1 is at origin.
    2. Scale by the distance between two reference points (e.g., outer corners of eyes).
    
    Modified to use only X and Y coordinates.
    """
    coords = []
    for idx in SELECTED_LANDMARKS:
        try:
            lm = landmarks[idx]
            x, y = lm.x, lm.y  # Removed z coordinate
            coords.append([x, y])
        except IndexError:
            coords.append([0, 0])  # Handle missing landmarks with zeros for X and Y
    coords = np.array(coords)

    # Reference points for normalization
    try:
        nose_ref = coords[SELECTED_LANDMARKS.index(1)]
        left_eye_corner = coords[SELECTED_LANDMARKS.index(33)]
        right_eye_corner = coords[SELECTED_LANDMARKS.index(263)]
        face_width = np.linalg.norm(left_eye_corner - right_eye_corner)
        if face_width < 1e-7:
            face_width = 1.0

        # Translate by nose_ref and scale
        coords = (coords - nose_ref) / face_width

        # Flatten to a single vector
        return coords.flatten()
    except (IndexError, ValueError):
        return np.zeros(len(SELECTED_LANDMARKS)*2)  # Adjusted for X and Y only

def collect_data_for_label(cap, label):
    """Collect data for a specific label/direction for CALIBRATION_TIME seconds."""
    data_points = []
    start_time = time.time()
    while True:
        ret, camera_frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break
        camera_frame = cv2.flip(camera_frame, 1)
        rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Create a display frame with solid background
        display_frame = np.zeros((480,640,3), dtype=np.uint8)
        display_frame[:] = COLORS[label]

        elapsed = time.time() - start_time
        remaining = CALIBRATION_TIME - elapsed
        cv2.putText(display_frame, f"Look {label}. Collecting... {remaining:.1f}s", 
                    (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Collect landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                eye_vector = normalize_landmarks(face_landmarks.landmark, camera_frame.shape)
                data_points.append(eye_vector)

        cv2.imshow("Calibration", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            print("Calibration interrupted by user.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

        if elapsed >= CALIBRATION_TIME:
            break

    return np.array(data_points)

def build_model(input_dim, num_classes):
    """Builds and compiles the neural network model."""
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def live_prediction(model):
    """
    Performs live prediction using the trained model.
    Displays the predicted direction on the video feed.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera for live prediction.")
        sys.exit(1)
    
    print("\nStarting live prediction. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                eye_vector = normalize_landmarks(face_landmarks.landmark, frame.shape)
                eye_vector = eye_vector.reshape(1, -1)
                prediction = model.predict(eye_vector)
                predicted_label = LABELS[np.argmax(prediction)]
                confidence = np.max(prediction)

                # Display the prediction on the frame
                cv2.putText(frame, f"Prediction: {predicted_label} ({confidence*100:.2f}%)",
                            (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, "No face detected.", 
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Live Prediction", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Live prediction terminated by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # Initialize or retrieve calibration data
    all_data = []
    all_labels = []

    # CALIBRATION PHASE
    print("\nStarting calibration...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        sys.exit(1)

    for idx, direction in enumerate(LABELS):
        # Show buffer screen for all directions except the very first (CENTER)
        if idx > 0:
            print(f"\nPrepare to look {direction} in {BUFFER_TIME} seconds...")
            buffer_start = time.time()
            while True:
                elapsed = time.time() - buffer_start
                if elapsed >= BUFFER_TIME:
                    break
                remaining = BUFFER_TIME - elapsed
                buffer_frame = np.zeros((480,640,3), dtype=np.uint8)
                buffer_frame[:] = COLORS["BUFFER"]
                cv2.putText(buffer_frame, f"Get ready to look {direction}", 
                            (30,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(buffer_frame, f"Starting in: {remaining:.1f}s", 
                            (30,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.imshow("Calibration", buffer_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("Calibration interrupted by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit()

        print(f"\nPlease look {direction} for {CALIBRATION_TIME} seconds.")
        data = collect_data_for_label(cap, direction)
        labels = np.full((data.shape[0],), idx)
        all_data.append(data)
        all_labels.append(labels)

    cap.release()
    cv2.destroyAllWindows()
    print("Calibration completed.")

    # Combine all collected data
    X = np.vstack(all_data)
    y = np.hstack(all_labels)

    # One-hot encode labels
    y_cat = to_categorical(y, num_classes=NUM_CLASSES)

    # BUILD & TRAIN MODEL
    print("\nTraining the model...")
    input_dim = X.shape[1]
    model = build_model(input_dim, NUM_CLASSES)
    model.fit(X, y_cat, epochs=EPOCHS, batch_size=16, verbose=1)
    print("Model trained successfully.")

    # Evaluate the model
    loss, accuracy = model.evaluate(X, y_cat, verbose=0)
    print(f"Training Accuracy: {accuracy*100:.2f}%")

    # LIVE INFERENCE
    live_prediction(model)

    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    main()
