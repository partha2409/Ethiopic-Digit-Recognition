import numpy as np
import os
import cv2


def load_training_data(train_path):
    train_images = np.zeros([50000, 28, 28])
    train_labels = np.zeros([50000])
    train_folders = os.listdir(train_path)
    row = 0
    for folder in train_folders:
        print("Loading training images folder {}...".format(folder))
        img_files = os.listdir(train_path + folder)
        for img in img_files:
            image = cv2.imread(train_path + folder + "/" + img, cv2.IMREAD_GRAYSCALE)
            train_images[row, :, :] = image
            train_labels[row] = int(folder) - 1
            row += 1
    return train_images, train_labels


def load_validation_data(val_path):
    val_images = np.zeros([10000, 28, 28])
    val_labels = np.zeros([10000])
    val_folders = os.listdir(val_path)
    row = 0
    for folder in val_folders:
        print("Loading validation images folder {}...".format(folder))
        img_files = os.listdir(val_path + folder)
        for img in img_files:
            image = cv2.imread(val_path + folder + "/" + img, cv2.IMREAD_GRAYSCALE)
            val_images[row, :, :] = image
            val_labels[row] = int(folder) - 1
            row += 1
    return val_images, val_labels


def load_test_data(test_path):
    test_images = np.zeros([10000, 28, 28])
    img_files = os.listdir(test_path)
    row = 0
    for img in img_files:
        image = cv2.imread(test_path + img, cv2.IMREAD_GRAYSCALE)
        test_images[row, :, :] = image
        row += 1
    return test_images


def classification_accuracy(logits, ground_truth):
    n_images = logits.shape[0]
    prediction = np.argmax(logits, axis=1)
    x = prediction - ground_truth
    n_wrong_predictions = np.count_nonzero(x)
    accuracy = (n_images - n_wrong_predictions) / n_images

    return accuracy
