import numpy as np
from PIL import Image
import os

PATH_TO_VAL_ANNOT = '/Users/leonidas/PycharmProjects/DL_project/data/tiny-imagenet-200/val/val_annotations.txt'
SAVE_PATH = '/Users/leonidas/PycharmProjects/DL_project/datasets/tiny_imagenet_data.npz'

def get_val_ids():
    lines = open(PATH_TO_VAL_ANNOT, 'r').read()
    val_annotations = {}
    for line in lines.splitlines():
        line_split = line.strip().split()
        val_annotations[line_split[0]] = line_split[1]

    return val_annotations


def read_image(path):
    image = np.array(Image.open(path))

    if len(np.shape(image)) == 2:
        image = np.array([image, image, image])
        image = np.transpose(image, (1, 2, 0))

    # else:
    #     image = np.transpose(X, (2, 0, 1))

    return image

def save_to_file(X_train, y_train, X_val, y_val, X_test):

    np.savez(SAVE_PATH, X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val, X_test=X_test)


def load_from_file(path):
    with np.load(path) as f:
        X_train, y_train = f['X_train'], f['y_train']
        X_val, y_val = f['X_val'], f['y_val']
        X_test = f['X_test']

    return [X_train, y_train, X_val, y_val, X_test]

def load_tiny_imagenet(path):
    # Load images
    num_classes = 200

    X_train = np.zeros([num_classes * 500, 64, 64, 3], dtype='uint8')
    y_train = np.zeros([num_classes * 500], dtype='uint8')
    X_val = np.zeros([num_classes * 50, 64, 64, 3], dtype='uint8')
    y_val = np.zeros([num_classes * 50], dtype='uint8')
    X_test = np.zeros([num_classes * 50, 64, 64, 3], dtype='uint8')

    train_path = path + '/train'
    val_path = path + '/val/images'
    test_path = path + '/test/images'
    val_map = get_val_ids()

    image_counter = 0
    class_index = 0

    classes = {}

    train_folders = os.listdir(train_path)
    train_folders.sort()
    train_folders.remove('.DS_Store')
    for root in train_folders:

        child_root = os.path.join(os.path.join(train_path, root), 'images')
        classes[root] = class_index
        for image in os.listdir(child_root):
            if not image.lower().endswith(".jpeg"):
                print('continue train {}'.format(image))
                continue

            X_train[image_counter] = read_image(os.path.join(child_root, image))
            y_train[image_counter] = class_index

            image_counter += 1
        class_index += 1

    image_counter = 0
    for image_filename in os.listdir(val_path):
        if not val_map[image_filename] in classes.keys():
            continue

        X_val[image_counter] = read_image(os.path.join(val_path, image_filename))
        y_val[image_counter] = classes[val_map[image_filename]]
        image_counter += 1

    for index, filename in enumerate(os.listdir(test_path)):
        if not filename.lower().endswith(".jpeg"):
            print('continue test {}'.format(filename))
            continue
        X_test[index] = read_image(os.path.join(test_path, filename))

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_val = np.reshape(y_val, (len(y_val), 1))

    save_to_file(X_train, y_train, X_val, y_val, X_test)

    return [X_train, y_train, X_val, y_val, X_test]

# Tiny ImageNet Challenge is the default course project for Stanford CS231N.
# It runs similar to the ImageNet challenge (ILSVRC).
# Tiny ImageNet has 200 classes and each class has 500 training images, 50 validation images, and 50 test images.
# The images are down-sampled to 64 x 64 pixels.
if __name__ == "__main__":
    path = '../data/tiny-imagenet-200'
    save_path = 'tiny_imagenet_data.npz'

    load_tiny_imagenet(path)

    npzfile = np.load(save_path)

    print(npzfile.files)

