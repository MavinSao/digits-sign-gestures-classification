import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras

# Load the trained digit classification model
model = keras.models.load_model('digit-classification-model')

# Define the labels for the 10 digit classes
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box coordinates
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

            # Increase the size of the bounding box
            scale_factor = 1.5  # Adjust the scale factor as needed
            new_w = int((x_max - x_min) * scale_factor)
            new_h = int((y_max - y_min) * scale_factor)
            new_x = max(0, x_min - (new_w - (x_max - x_min)) // 2)
            new_y = max(0, y_min - (new_h - (y_max - y_min)) // 2)

            # Draw a rectangle around the hand
            cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (255, 0, 0), 2)

            # Crop the hand region
            hand_crop = frame[new_y:new_y + new_h, new_x:new_x + new_w]

            if hand_crop.size > 0:
                # Resize the cropped hand to 100x100
                hand_crop = cv2.resize(hand_crop, (100, 100))

                # Expand dimensions to match the input shape of the model
                hand_crop = np.expand_dims(hand_crop, axis=0)  # Add batch dimension

                # Ensure the image is in float format
                hand_crop = hand_crop.astype('float32') / 255.0

                # Make a prediction
                predict = model.predict(hand_crop)
                predict_number = np.argmax(predict)
                confidence_score = np.max(predict)

                # Get the predicted label
                predicted_label = labels[predict_number]

                # Print the predicted label and confidence score
                print(f"{predicted_label}: {confidence_score:.2f}")

                # Display the prediction on the frame
                text = f"{predicted_label}: {confidence_score:.2f}"
                cv2.putText(frame, text, (new_x, new_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame with the hand detection box and prediction
    cv2.imshow('cam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
