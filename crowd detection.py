import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, SimpleRNN
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# Load YOLOv8 Model
yolo_model = YOLO("yolov8n.pt")  

# Load MediaPipe for Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def capture_video(video_source=0):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return None
    return cap

def create_cnn_classifier(input_shape):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_feature_extractor():
    base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    model = Model(inputs=base_model.input, outputs=Flatten()(base_model.output))
    return model

def load_frame_dataset(dataset_path, sample_limit=None):
    frames, labels = [], []
    class_dirs = {'train': 1, 'valid': 0}

    for class_name, label in class_dirs.items():
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            continue

        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image = cv2.imread(file_path)
                if image is None:
                    continue
                image = cv2.resize(image, (128, 128)) / 255.0
                frames.append(image)
                labels.append(label)
            if sample_limit and len(frames) >= sample_limit:
                break

    return np.array(frames), np.array(labels)

def train_cnn_classifier(dataset_path):
    frames, labels = load_frame_dataset(dataset_path, sample_limit=5000)
    if len(frames) == 0:
        raise ValueError("No training data found. Check dataset path and structure.")

    X_train, X_val, y_train, y_val = train_test_split(frames, labels, test_size=0.2, random_state=42)
    model = create_cnn_classifier((128, 128, 3))

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, verbose=1)
    return model

def create_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(64, return_sequences=True, input_shape=input_shape),
        SimpleRNN(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def process_frame(frame, feature_extractor):
    frame_resized = cv2.resize(frame, (128, 128)) / 255.0
    return feature_extractor.predict(frame_resized[np.newaxis, ...], verbose=0)[0]

def detect_pose(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return results.pose_landmarks if results.pose_landmarks else None

def detect_objects(frame):
    results = yolo_model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            confidence = box.conf[0]  
            label = yolo_model.names[int(box.cls[0])]  

            if confidence > 0.5:  
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def preprocess_dataset(dataset_path, feature_extractor):
    frames, labels = load_frame_dataset(dataset_path, sample_limit=5000)
    features = np.array([feature_extractor.predict(frame[np.newaxis, ...], verbose=0)[0] for frame in frames])
    
    sequence_length = 30
    sequences, seq_labels = [], []
    for i in range(len(features) - sequence_length):
        sequences.append(features[i:i+sequence_length])
        seq_labels.append(labels[i+sequence_length])

    return np.array(sequences), np.array(seq_labels)

def alert_security(frame):
    cv2.putText(frame, "ALERT: Suspicious Activity!", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def main(video_source=0):
    cap = capture_video(video_source)
    if not cap:
        return
    
    feature_extractor = create_feature_extractor()
    rnn_model = create_rnn_model((30, feature_extractor.output_shape[1]))

    sequence = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects(frame)
        pose_landmarks = detect_pose(frame)

        features = process_frame(frame, feature_extractor)
        sequence.append(features)
        
        if len(sequence) > 30:
            sequence.pop(0)

        if len(sequence) == 30:
            sequence_array = np.array(sequence)[np.newaxis, ...]
            prediction = rnn_model.predict(sequence_array, verbose=0)[0][0]
            
            if prediction > 0.5:
                alert_security(frame)
        
        cv2.imshow('Live Surveillance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\Mohammed Haris\OneDrive\Desktop\blahh\weapon-detection.v1i.createml"
    VIDEO_SOURCE = 0  

    print("Training CNN Classifier...")
    cnn_classifier = train_cnn_classifier(DATASET_PATH)
    feature_extractor = create_feature_extractor()

    print("Preparing RNN Data...")
    features, labels = preprocess_dataset(DATASET_PATH, feature_extractor)
    if len(features) == 0:
        raise ValueError("No valid sequences found for RNN training")

    print("Training RNN Model...")
    rnn_model = create_rnn_model((30, features.shape[2]))
    rnn_model.fit(features, labels, epochs=10, batch_size=8, validation_split=0.2, verbose=1)

    print("Starting Live Surveillance...")
    main(VIDEO_SOURCE)
