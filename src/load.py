import pandas as pd
import numpy as np
import os

# Load your training CSV file
training_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv'))
df = pd.read_csv(training_data_path)

# Normalize pixel values to [0,1]
pixels = df.drop(columns=['character']).values.astype(np.float32) / 255.0

# Convert 'digit_0', 'digit_1', ... labels to integers 0, 1, ...
def label_to_int(label):
    return int(label.split('_')[1])

labels = df['character'].map(label_to_int).values

# One-hot encode the labels to (10,1) vectors
def vectorize_label(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def get_training_data():
    # Create your training data list: each item is (input_vector, output_vector)
    training_data = []
    for i in range(len(pixels)):
        x = pixels[i].reshape((1024, 1))  # reshape to (1024,1)
        y = vectorize_label(labels[i])    # one-hot output vector (10,1)
        training_data.append((x, y))
    return training_data
