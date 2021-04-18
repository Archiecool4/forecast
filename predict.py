import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

from pix2pix import Generator, Discriminator, IMG_WIDTH, IMG_HEIGHT


def get_model():
    generator = Generator()
    discriminator = Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir = './training_checkpoints'
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    return generator


def predict_images(img, generator, years=5):
    imgs = []
    shape = (img.shape[1], img.shape[2])
    img = tf.image.resize(img[0], [IMG_HEIGHT, IMG_WIDTH],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = (img / 127.5) - 1
    img = np.expand_dims(img, axis=0)
    for _ in range(years):
        img = generator(img)

        resize_img = tf.image.resize(img[0] * 0.5 + 0.5, [shape[0], shape[1]],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        imgs.append(resize_img.numpy())
    return imgs
