"""
Author : Mavin Sao
Date : 2024.06.03.
"""
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Import the DataReader class from your module
from data_reader import DataReader, draw_graph

# Decide how many epochs to train for.
EPOCHS = 5  # Example default is 5

# Read the data.
dr = DataReader()
dr.read_images()

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen.fit(dr.train_X)

# Load the pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

def build_model(base_model, fine_tune_at):
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Add custom top layers
    model = keras.Sequential([
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')
    
    return model

# Fine-Tuning Level: Fix the lower 5 layers
model = build_model(base_model, fine_tune_at=5)

# Save the model summary to an image
plot_model(model, to_file='model_summary.png', show_shapes=True, show_layer_names=True)

# Train the model
print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(datagen.flow(dr.train_X, dr.train_Y, batch_size=64),
                    validation_data=(dr.valid_X, dr.valid_Y),
                    epochs=EPOCHS,
                    callbacks=[early_stop])

# Evaluate and print accuracy table for each class
def evaluate_model(model, test_X, test_Y, model_name):
    predictions = model.predict(test_X)
    y_pred = np.argmax(predictions, axis=1)
    report = classification_report(test_Y, y_pred, output_dict=True, target_names=[str(i) for i in range(10)])
    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(test_Y, y_pred, target_names=[str(i) for i in range(10)]))
    return report

report = evaluate_model(model, dr.test_X, dr.test_Y, "Fine-Tuning")

# Save the model
model.save('digit-classification-model')

# Plot and save the training results
draw_graph(history)

# Save accuracy score images for model's evaluation metrics
def save_accuracy_score_image(report, model_name):
    classes = list(report.keys())[:-3]
    accuracies = [report[cls]['precision'] for cls in classes]
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, accuracies, color='blue')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Scores for {model_name}')
    plt.ylim(0, 1)
    plt.savefig(f'{model_name}_accuracy_scores.png')
    plt.close()

save_accuracy_score_image(report, "Fine-Tuning")
