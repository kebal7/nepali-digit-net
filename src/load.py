import pandas as pd
import numpy as np
import os

def label_to_int(label):
    return int(label.split('_')[1])

def vectorize_label(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data(csv_path, one_hot=True):
    df = pd.read_csv(csv_path)

    pixels = df.drop(columns=['character']).values.astype(np.float32) / 255.0
    labels = df['character'].map(label_to_int).values

    data = []
    for i in range(len(pixels)):
        x = pixels[i].reshape((1024, 1))
        y = vectorize_label(labels[i]) if one_hot else labels[i]
        data.append((x, y))
    return data

def get_training_and_test_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    train_csv = os.path.join(base_dir, 'train.csv')
    test_csv = os.path.join(base_dir, 'test.csv')

    training_data = load_data(train_csv, one_hot=True)
    test_data = load_data(test_csv, one_hot=False)
    
    return training_data, test_data

