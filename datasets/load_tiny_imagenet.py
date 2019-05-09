import numpy as np
from PIL import Image
import os


def get_val_ids():

    valAnnotationsContents = open('../data/tiny-imagenet-200/val/val_annotations.txt', 'r').read()
    valAnnotations = {}
    for line in valAnnotationsContents.splitlines():
        line_split = line.strip().split()
        valAnnotations[line_split[0]] = line_split[1]

    return valAnnotations


def load_tiny_imagenet(path):

    # Load images
    num_classes = 200

    X_train = np.zeros([num_classes * 500, 3, 64, 64], dtype='uint8')
    y_train = np.zeros([num_classes * 500], dtype='uint8')
    X_test = np.zeros([num_classes * 50, 3, 64, 64], dtype='uint8')
    y_test = np.zeros([num_classes * 50], dtype='uint8')

    trainPath = path + '/train'
    testPath = path + '/val/images'

    val_map = get_val_ids()

    image_counter = 0
    class_index = 0

    classes = {}

    train_folders = os.listdir(trainPath)
    train_folders.sort()
    train_folders.remove('.DS_Store')
    for root in train_folders:
        child_root = os.path.join(os.path.join(trainPath, root), 'images')
        classes[root] = class_index
        for image in os.listdir(child_root):
            X = np.array(Image.open(os.path.join(child_root, image)))

            if len(np.shape(X)) == 2:
                X_train[image_counter] = np.array([X, X, X])
            else:
                X_train[image_counter] = np.transpose(X, (2, 0, 1))

            y_train[image_counter] = class_index
            image_counter += 1
        class_index += 1

    image_counter = 0
    for root in os.listdir(testPath):
        if not val_map[root] in classes.keys():
            continue

        child_root = os.path.join(testPath, root)
        X = np.array(Image.open(child_root))
        if len(np.shape(X)) == 2:
            X_test[image_counter] = np.array([X, X, X])
        else:
            X_test[image_counter] = np.transpose(X, (2, 0, 1))
        y_test[image_counter] = classes[val_map[root]]
        image_counter += 1

    return X_train, y_train, X_test, y_test


# Tiny ImageNet Challenge is the default course project for Stanford CS231N.
# It runs similar to the ImageNet challenge (ILSVRC).
# Tiny ImageNet has 200 classes and each class has 500 training images, 50 validation images, and 50 test images.
# The images are down-sampled to 64 x 64 pixels.
if __name__ == "__main__":

    path = '../data/tiny-imagenet-200'
    X_train, y_train, X_test, y_test = load_tiny_imagenet(path)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

