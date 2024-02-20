import time

import numpy as np
import cv2
import os


def load_mnist_dataset(dataset, path: str):
    print(f"Loading MNIST dataset {dataset}...")
    t1 = time.time()
    labels = os.listdir(os.path.join(path, dataset))

    X = []
    y = []

    for label in labels:

        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(
                os.path.join(path, dataset, label, file),
                cv2.IMREAD_UNCHANGED
            )

            X.append(image)
            y.append(int(label))
    t2 = time.time()
    print(f"Loaded {dataset} in {t2-t1} seconds.")

    return np.array(X), np.array(y).astype('uint8')


def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test
