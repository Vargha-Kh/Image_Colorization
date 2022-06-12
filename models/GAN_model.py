import tensorflow as tf
from tensorflow.keras.layers import (
    Activation, AveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose,
    Dense, Dropout, Flatten, Input, LeakyReLU, ReLU, UpSampling2D)
from tensorflow.keras.models import Sequential, Model


def make_generator_model():
    IMAGE_SIZE = 32
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(Conv2D(64, (3, 3), padding='same', strides=2, activation='relu'))
    model.add(BatchNormalization())
    # model = MaxPooling2D(pool_size=(2, 2))(model)

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', strides=2))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    # model = MaxPooling2D(pool_size=(2, 2))(model)

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(2, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    # self.model = BatchNormalization()(self.model)
    # self.model = merge(inputs=[self.g_input, self.model], mode='concat')
    # self.model = Activation('linear')(self.model)
    return model


def downsample(filters, kernel_size, apply_batchnorm=True):
    initializer = tf.random_uniform_initializer(0, 0.02)
    model = Sequential()
    model.add(Conv2D(filters, kernel_size, strides=2, padding='same',
                     kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        model.add(BatchNormalization())

    model.add(LeakyReLU())
    return model


def upsample(filters, kernel_size, apply_dropout=False):
    initializer = tf.random_uniform_initializer(0, 0.02)
    model = Sequential()
    model.add(Conv2DTranspose(filters, kernel_size, strides=2, padding='same',
                              kernel_initializer=initializer, use_bias=False))
    model.add(BatchNormalization())

    if apply_dropout:
        model.add(Dropout(0.5))

    model.add(ReLU())
    return model


def make_autoencoder_generator_model():
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

    # Downsampling layers
    # 1: (BATCH_SIZE, 16, 16, 32)
    # 2: (BATCH_SIZE, 8, 8, 64)
    # 3: (BATCH_SIZE, 4, 4, 128)
    # 4: (BATCH_SIZE, 2, 2, 256)
    # 5: (BATCH_SIZE, 1, 1, 256)

    downstack = [
        downsample(32, 4, apply_batchnorm=False),
        downsample(64, 4),
        downsample(128, 4),
        downsample(256, 4),
        downsample(256, 4)
    ]

    # Upsampling layers
    # 1: (BATCH_SIZE, 1, 1, 256)
    # 2: (BATCH_SIZE, 1, 1, 128)
    # 3: (BATCH_SIZE, 1, 1, 64)
    # 4: (BATCH_SIZE, 1, 1, 32)

    upstack = [
        upsample(256, 4, apply_dropout=True),
        upsample(128, 4),
        upsample(64, 4),
        upsample(32, 4),
    ]

    initializer = tf.random_uniform_initializer(0, 0.02)
    output_layer = Conv2DTranspose(2, 3, strides=2, padding='same',
                                   kernel_initializer=initializer,
                                   activation='tanh')

    x = inputs

    # Downsampling layers
    skips = []
    for dm in downstack:
        x = dm(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling layers
    for um, skip in zip(upstack, skips):
        x = um(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = output_layer(x)

    return Model(inputs=inputs, outputs=x)


def make_discriminator_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(.2))
    model.add(BatchNormalization())
    model.add(Dropout(.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


LAMBDA = 100
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss
