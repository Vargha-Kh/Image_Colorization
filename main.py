from data_preprocessing import load_data
from time import time
import os
import tensorflow as tf
import numpy as np
from training import checkpoints, train_step
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
import sys

IMAGE_SIZE = 32
EPOCHS = 150
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 100


def main():
    WORKDIR = "./"
    home = "../input/image-colorization"
    os.listdir("../input/image-colorization/ab/ab")
    X_train, Y_train, X_test, Y_test = load_data(home, channels_first=True)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    (generator, discriminator, generator_optimizer,
     discriminator_optimizer), summary_writer, checkpoint, checkpoint_prefix = checkpoints()
    for e in range(EPOCHS):
        start_time = time()
        gen_loss_total = disc_loss_total = 0
        for input_image, target in train_dataset:
            gen_loss, disc_loss = train_step(input_image, target, e)
            gen_loss_total += gen_loss
            disc_loss_total += disc_loss

        time_taken = time() - start_time

        if (e + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {}: gen loss: {}, disc loss: {}, time: {:.2f}s'.format(
            e + 1, gen_loss_total / BATCH_SIZE, disc_loss_total / BATCH_SIZE, time_taken))

    Y_hat = generator(X_test[:250])
    total_count = len(Y_hat)

    for idx, (x, y, y_hat) in enumerate(zip(X_test[:250], Y_test[:250], Y_hat)):
        # Original RGB image
        orig_lab = np.dstack((x, y * 128))
        orig_rgb = lab2rgb(orig_lab)

        # Grayscale version of the original image
        grayscale_lab = np.dstack((x, np.zeros((IMAGE_SIZE, IMAGE_SIZE, 2))))
        grayscale_rgb = lab2rgb(grayscale_lab)

        # Colorized image
        predicted_lab = np.dstack((x, y_hat * 128))
        predicted_rgb = lab2rgb(predicted_lab)

        # print(idx)

        # fig = plt.figure()
        # fig.add_subplot(1, 3, 1)
        # plt.axis('off')
        # plt.imshow(grayscale_rgb)
        # fig.add_subplot(1, 3, 2)
        # plt.axis('off')
        # plt.imshow(orig_rgb)
        # fig.add_subplot(1, 3, 3)
        # plt.axis('off')
        # plt.imshow(predicted_rgb)
        # plt.show()

        plt.axis('off')
        plt.imshow(grayscale_rgb)
        plt.savefig(os.path.join(WORKDIR, 'results', '{}-bw.png'.format(idx)), dpi=1)

        plt.axis('off')
        plt.imshow(orig_rgb)
        plt.savefig(os.path.join(WORKDIR, 'results', '{}-gt.png'.format(idx)), dpi=1)

        plt.axis('off')
        plt.imshow(predicted_rgb)
        plt.savefig(os.path.join(WORKDIR, 'results', '{}-gan.png'.format(idx)), dpi=1)

        sys.stdout.flush()
        sys.stdout.write('\r{} / {}'.format(idx + 1, total_count))

    tf.saved_model.save(generator, os.path.join(WORKDIR, "generator-saved-model"))
    tf.saved_model.save(discriminator, os.path.join(WORKDIR, "disciminator-saved-model"))
