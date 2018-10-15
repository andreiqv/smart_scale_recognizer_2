import re

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import math

# tfe = tf.contrib.eager
# tf.enable_eager_execution()
# slim = tf.contrib.slim


def plot_random_nine(images, labels, names):
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    idx = np.arange(0, int(images.shape[0]))
    np.random.shuffle(idx)
    idx = idx[:9]

    for i, ax in enumerate(axes.flat):
        original = images[idx[i]]
        label = np.argmax(labels[idx[i], :])

        np_image = np.uint8(original * 255)  # [..., [0,1,2]]
        im = Image.fromarray(np_image).resize((140, 120), Image.BILINEAR)
        draw = ImageDraw.Draw(im)
        draw.text((10, 10), names[label], fill=(255, 255, 255, 128))
        del draw

        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


class GoodsDataset:
    def __init__(self, path_list,
                 labels_list,
                 image_size,
                 train_batch,
                 valid_batch,
                 multiply,
                 valid_percentage) -> None:
        super().__init__()
        self.path_list = path_list
        self.image_size = image_size
        self.labels_list = labels_list
        self.train_batch = train_batch
        self.valid_batch = valid_batch
        self.multiply = multiply
        self.valid_percentage = valid_percentage

        self.load_images()

    def load_images(self):
        train_image_paths = np.array([])
        valid_image_paths = np.array([])

        train_image_labels = []
        valid_image_labels = []

        images_dict = {}

        with open(self.path_list, "r") as pl:
            for line in pl:
                line = line.strip()
                line = line.replace("\n", "")
                plu_id = line.split("/")[-2]

                if plu_id not in images_dict:
                    images_dict[plu_id] = [line]
                else:
                    images_dict[plu_id].append(line)

        self.classes_count = len(images_dict.keys())

        for index, plu_id in enumerate(images_dict.keys()):
            images_dict[plu_id] = np.array(images_dict[plu_id])
            valid_mask = np.zeros(len(images_dict[plu_id]), dtype=bool)
            one_hot = np.eye(self.classes_count, self.classes_count)[index]
            one_hot = np.tile(one_hot, (len(images_dict[plu_id]), 1))

            if len(images_dict[plu_id]) >= 20:
                idx_count = int(len(images_dict[plu_id]) * self.valid_percentage)
                idx = np.arange(len(images_dict[plu_id]))
                np.random.shuffle(idx)
                idxs = idx[:idx_count]

                valid_mask[idxs] = True
                valid_images = images_dict[plu_id][valid_mask]

                valid_image_paths = np.append(valid_image_paths, valid_images)
                if valid_image_labels == []:
                    valid_image_labels = one_hot[valid_mask]
                else:
                    valid_image_labels = np.concatenate([valid_image_labels, one_hot[valid_mask]])

            if train_image_labels == []:
                train_image_labels = one_hot[~valid_mask]
            else:
                train_image_labels = np.concatenate([train_image_labels, one_hot[~valid_mask]])
            train_image_paths = np.append(train_image_paths, images_dict[plu_id][~valid_mask])

        with open(self.labels_list, "w") as l_f:
            for index, plu_id in enumerate(images_dict.keys()):
                l_f.write(plu_id + "\n")

        print("train dataset", train_image_labels.shape[0])
        print("valid dataset", valid_image_labels.shape[0])
        randomize = np.arange(valid_image_labels.shape[0])
        np.random.shuffle(randomize)

        self.valid_image_labels = valid_image_labels[randomize]
        self.valid_image_paths = valid_image_paths[randomize]

        randomize = np.arange(train_image_labels.shape[0])
        np.random.shuffle(randomize)

        self.train_image_labels = train_image_labels[randomize]
        self.train_image_paths = train_image_paths[randomize]

    def _parse_function(self, image_path, label):
        image_string = tf.read_file(image_path)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [self.image_size[1], self.image_size[0]],
                                               method=tf.image.ResizeMethod.BICUBIC)
        image = tf.cast(image_resized, tf.float32) / tf.constant(255.0)

        return image, label,

    def _augment_dataset(self, dataset, multiply, batch):
        dataset = dataset.repeat().batch(batch)

        def _random_distord(images, labels):
            images = tf.image.random_flip_left_right(images)
            images = tf.image.random_flip_up_down(images)

            # angle = tf.random_uniform(shape=(1,), minval=0, maxval=90)
            # images = tf.contrib.image.rotate(images, angle * math.pi / 180, interpolation='BILINEAR')

            images = tf.image.random_hue(images, max_delta=0.05)
            images = tf.image.random_contrast(images, lower=0.9, upper=1.5)
            images = tf.image.random_brightness(images, max_delta=0.1)
            images = tf.image.random_saturation(images, lower=1.0, upper=1.5)

            images = tf.minimum(images, 1.0)
            images = tf.maximum(images, 0.0)
            images.set_shape([None, None, None, 3])
            return images, labels

        dataset = dataset.map(_random_distord)

        return dataset

    def get_train_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.train_image_paths, self.train_image_labels))
        dataset = dataset.map(self._parse_function)

        return self._augment_dataset(dataset, self.multiply, self.train_batch)

    def get_valid_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.valid_image_paths, self.valid_image_labels))
        dataset = dataset.map(self._parse_function)

        return dataset.batch(self.valid_batch)


if __name__ == '__main__':
    lbls_file = "output/labels.txt"
    goods_dataset = GoodsDataset("dataset.list", lbls_file, (299, 299), 16, 16, 2, 0.1)
    names = []
    with open(lbls_file, "r") as l_f:
        for line in l_f:
            names.append(line.replace("\n", ""))
    for i, (images, labels) in enumerate(goods_dataset.get_valid_dataset()):
        plot_random_nine(images, labels, names)
