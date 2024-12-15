import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Configuration (should match the training script)
RIGHT_EYE_REGION = [33, 133, 160, 158, 159, 144, 145, 153, 246]
LEFT_EYE_REGION = [263, 362, 387, 385, 380, 373, 374, 381, 382]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474, 475, 476, 477]
NOSE_LANDMARKS = [1, 4]

SELECTED_LANDMARKS = RIGHT_EYE_REGION + LEFT_EYE_REGION + RIGHT_IRIS + LEFT_IRIS + NOSE_LANDMARKS
LABELS = ["CENTER", "UP", "LEFT", "DOWN", "RIGHT"]

def normalize_landmarks(landmarks, image_shape):
    """
    Extract selected landmarks, then normalize:
    1. Translate so that nose landmark #1 is at origin.
    2. Scale by the distance between two reference points (e.g., outer corners of eyes).
    """
    coords = []
    for idx in SELECTED_LANDMARKS:
        try:
            lm = landmarks[idx]
            x, y, z = lm.x, lm.y, lm.z
            coords.append([x, y, z])
        except IndexError:
            coords.append([0, 0, 0])  # Handle missing landmarks
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
        return np.zeros(len(SELECTED_LANDMARKS)*3)

def main():
    # Path to your downloaded TFLite model
    tflite_model_path = 'gaze_model.tflite'

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Open camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process face landmarks
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Normalize landmarks
                input_vector = normalize_landmarks(face_landmarks.landmark, frame.shape)

                # Prepare input for TFLite model
                input_data = np.array([input_vector], dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)

                # Run inference
                interpreter.invoke()

                # Get the output
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predicted_class = np.argmax(output_data)

                # Display prediction
                prediction_text = LABELS[predicted_class]
                cv2.putText(frame, f"Gaze: {prediction_text}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Gaze Prediction', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()