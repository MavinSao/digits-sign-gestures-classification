import cv2
import numpy as np
from tensorflow import keras


def load_model(model_path='digit-classification-model'):
    # Load the trained model
    model = keras.models.load_model(model_path)
    return model


def preprocess_image(image_path, target_size=(100, 100)):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read")

    # Resize the image
    image = cv2.resize(image, target_size)

    # Convert the image to RGB (if needed)
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Normalize the image
    image = image.astype('float32') / 255.0

    # Expand dimensions to match the input shape of the model
    image = np.expand_dims(image, axis=0)

    return image


def predict_image(model, image_path):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(image)
    predict_number = np.argmax(prediction)
    confidence_score = np.max(prediction)

    return predict_number, confidence_score


def get_label(predicted_number, labels=None):
    if labels is None:
        labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return labels[predicted_number]


# Example usage
if __name__ == "__main__":

    model_path = 'digit-classification-model'
    image_path = 'data/test/2/IMG_0603.JPG'

    model = load_model(model_path)
    predicted_number, confidence_score = predict_image(model, image_path)
    predicted_label = get_label(predicted_number)

    print(f"Predicted Label: {predicted_label}, Confidence Score: {confidence_score:.2f}")
