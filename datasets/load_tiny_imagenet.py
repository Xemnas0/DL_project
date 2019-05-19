import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from tqdm import tqdm
import os
import zipfile

from nnabla.logger import logger
from nnabla.utils.data_source_loader import download

PATH_TO_VAL_ANNOT = 'datasets/tiny-imagenet-200/val/val_annotations.txt'
SAVE_PATH = 'datasets/tiny_imagenet_data.npz'


def download_tiny_imagenet():
    """
        donwload and unzip the tiny imagenet dataset

        Arguments:
            dataset_name: can be MNIST, CIFAR10, CIFAR100 or TINY_IMAGENET.

        Returns:
                dir_data: the directory of the data containing the tiny imagenet
    """

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    dir_data = os.path.join('datasets', 'tiny-imagenet-200')
    if not os.path.isdir(dir_data):
        f = download(url)
        logger.info('Extracting {} ...'.format(f.name))
        z = zipfile.ZipFile(f)
        d = 'datasets/'
        l = z.namelist()
        for i in tqdm(range(len(l))):
            z.extract(l[i], d)
        z.close()
        f.close()
    return dir_data


def get_val_ids():
    """
        get validation images and corresponding IDs from file


        Returns:
                val_annotations: dictionary with ID value and image name key
    """

    lines = open(PATH_TO_VAL_ANNOT, 'r').read()
    val_annotations = {}
    for line in lines.splitlines():
        line_split = line.strip().split()
        val_annotations[line_split[0]] = line_split[1]

    return val_annotations


def read_image(path):
    """
        read image data. if it has one channel copy so that it has 3 channels

        Arguments:
            path: path to image

        Returns:
                image: array with image values
    """

    image = np.array(Image.open(path))

    if len(np.shape(image)) == 2:
        image = np.array([image, image, image])
        image = np.transpose(image, (1, 2, 0))

    return image


def save_to_file(X_train, y_train, X_val, y_val):
    np.savez(SAVE_PATH, X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val)


def load_from_file(path):
    with np.load(path) as f:
        X_train, y_train = f['X_train'], f['y_train']
        X_val, y_val = f['X_val'], f['y_val']

    return [X_train, y_train, X_val, y_val]


def load_tiny_imagenet(path):
    """
          loads tiny imagenet

           Arguments:
               path: path to tiny imagenet

           Returns:
                   X_train: array with train data
                   y_train: array with train labels
                   X_val: array with validation data
                   y_val: array with validation labels
       """
    num_classes = 200

    X_train = np.zeros([num_classes * 500, 64, 64, 3], dtype='uint8')
    y_train = np.zeros([num_classes * 500], dtype='uint8')
    X_val = np.zeros([num_classes * 50, 64, 64, 3], dtype='uint8')
    y_val = np.zeros([num_classes * 50], dtype='uint8')

    train_path = path + '/train'
    val_path = path + '/val/images'
    val_map = get_val_ids()

    image_counter = 0
    class_index = 0

    classes = {}

    train_folders = os.listdir(train_path)
    train_folders.sort()

    # mac compatibility
    folder = '.DS_Store'
    if folder in train_folders:
        train_folders.remove(folder)
    print('Loading train set')
    for root in tqdm(train_folders):

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
    print('Loading val set')
    for image_filename in tqdm(os.listdir(val_path)):
        if not val_map[image_filename] in classes.keys():
            continue

        X_val[image_counter] = read_image(os.path.join(val_path, image_filename))
        y_val[image_counter] = classes[val_map[image_filename]]
        image_counter += 1

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_val = np.reshape(y_val, (len(y_val), 1))

    X_train, y_train = shuffle(X_train, y_train)

    save_to_file(X_train, y_train, X_val, y_val)

    return [X_train, y_train, X_val, y_val]


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
