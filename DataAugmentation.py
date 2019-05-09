import matplotlib.pyplot as plt
import tensorflow as tf

class DataAugmentation:


    def __init__(self):
        pass

    def crop(self, image, central_fraction=0.7):
        return tf.image.central_crop(image,central_fraction=central_fraction)

    def flip_left_right(self, image):
        return tf.image.flip_left_right(image)

    def flip_upside_down(self, image):
        return tf.image.flip_up_down(image)

    def pad_and_crop(self, image, shape, pad_size=2):
        image = tf.image.pad_to_bounding_box(image, pad_size, pad_size, shape[0] + pad_size * 2,
                                             shape[1] + pad_size * 2)
        return tf.image.random_crop(image, shape)

    def standardization(self, image):
        return tf.image.per_image_standardization(image)



def show_image(original_image, augmented_image, title):
    fig = plt.figure()
    fig.suptitle(title)

    original_plt = fig.add_subplot(1, 2, 1)

    original_plt.set_title('original image')
    original_plt.imshow(original_image)

    augmented_plt = fig.add_subplot(1, 2, 2)
    augmented_plt.set_title('augmented image')
    augmented_plt.imshow(augmented_image)
    plt.show(block=True)


if __name__ == "__main__":

    img_path = 'data/tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG'

    img_raw = tf.io.read_file(img_path)
    img_tensor = tf.image.decode_image(img_raw)

    augment = DataAugmentation()

    cropped = augment.crop(img_tensor, central_fraction=0.7)
    show_image(img_tensor, cropped, 'cropped image')

    flipped = augment.flip_left_right(img_tensor)
    show_image(img_tensor, flipped, 'flip_image_left_right')

    flipped = augment.flip_upside_down(img_tensor)
    show_image(img_tensor, flipped, 'flip_image_upside_down')

    # pad = augment.pad_and_crop(img_tensor, [30,20])
    # show_image(img_tensor, pad, 'pad')

    # stand = augment.standardization(img_tensor)
    # show_image(img_tensor, stand, 'standardization')

