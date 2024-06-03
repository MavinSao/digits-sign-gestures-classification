"""
Author : Mavin Sao
Date : 2024.06.03.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

class DataReader():
    def __init__(self):
        self.label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.train_X = []
        self.train_Y = []
        self.valid_X = []
        self.valid_Y = []
        self.test_X = []
        self.test_Y = []

    def read_images(self):
        print("Reading Data...")
        self.train_X, self.train_Y = self.load_data("data/train")
        self.valid_X, self.valid_Y = self.load_data("data/valid")
        self.test_X, self.test_Y = self.load_data("data/test")

        self.train_X = np.asarray(self.train_X) / 255.0
        self.train_Y = np.asarray(self.train_Y)

        self.valid_X = np.asarray(self.valid_X) / 255.0
        self.valid_Y = np.asarray(self.valid_Y)

        self.test_X = np.asarray(self.test_X) / 255.0
        self.test_Y = np.asarray(self.test_Y)

        # Print the information of the read data.
        print("\n\nData Read Done!")
        print("Training X Size : " + str(self.train_X.shape))
        print("Training Y Size : " + str(self.train_Y.shape))
        print("Validate X Size : " + str(self.valid_X.shape))
        print("Validate Y Size : " + str(self.valid_Y.shape))
        print("Test X Size : " + str(self.test_X.shape))
        print("Test Y Size : " + str(self.test_Y.shape) + '\n\n')

    def load_data(self, directory):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory '{directory}' does not exist.")
        
        data_x = []
        data_y = []

        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        for cls in classes:
            cls_path = os.path.join(directory, cls)
            print("Opening " + cls + "/")
            for el in os.listdir(cls_path):
                img_path = os.path.join(cls_path, el)
                image = Image.open(img_path).resize((100, 100))
                data_x.append(np.asarray(image))
                data_y.append(self.label.index(cls))
                image.close()
        return data_x, data_y

def draw_graph(history):
    train_history = history.history["loss"]
    validation_history = history.history["val_loss"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Loss History")
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS Function")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("train_history.png")

    train_history = history.history["accuracy"]
    validation_history = history.history["val_accuracy"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Accuracy History")
    plt.xlabel("EPOCH")
    plt.ylabel("Accuracy")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("accuracy_history.png")
