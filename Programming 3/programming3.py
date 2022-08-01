# Programming Assignment 3
# Logan Peticolas

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras.applications import InceptionResNetV2
from IPython.display import Image, display
import os


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
ds_dir = f'{ROOT_DIR}/Assets/dog vs cat'
pre_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150,150,3))



# tfds.disable_progress_bar()

# train_ds, validation_ds, test_ds = tfds.load(
#     "cats_vs_dogs",
#     # Reserve 10% for validation and 10% for test
#     split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
#     as_supervised=True,  # Include labels
# )

# tf.keras.preprocessing.image_dataset_from_directory(
#     directory,
#     labels="inferred",
#     label_mode="int",
#     class_names=None,
#     color_mode="rgb",
#     batch_size=32,
#     image_size=(256, 256),
#     shuffle=True,
#     seed=None,
#     validation_split=None,
#     subset=None,
#     interpolation="bilinear",
#     follow_links=False,
#     crop_to_aspect_ratio=False,
#     **kwargs
# )


# print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
# print(
#     "Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds)
# )
# print("Number of test samples: %d" % tf.data.experimental.cardinality(test_ds))


## https://keras.io/guides/transfer_learning/#standardizing-the-data ##
# Freeze the base_model
#pre_model.trainable = False

# Create new model on top
#inputs = keras.Input(shape=(150, 150, 3))

# Pre-trained Xception weights requires that input be scaled
# from (0, 255) to a range of (-1., +1.), the rescaling layer
# outputs: `(inputs * scale) + offset`
#scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
#x = scale_layer(x)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
# x = pre_model(x, training=False)
# x = keras.layers.GlobalAveragePooling2D()(x)
# x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
# outputs = keras.layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)

#model.summary()


#pre_model.summary()





## From https://keras.io/examples/vision/visualizing_what_convnets_learn/ ##
# Modified for 8x4 layout
def createFilterImages():
    # The dimensions of our input image
    img_width = 150
    img_height = 150

    # Our target layer: we will visualize the filters from this layer.
    # See `model.summary()` for list of layer names, if you want to change this.
    layer_name = "conv2d"


    # Set up a model that returns the activation values for our target layer
    layer = pre_model.get_layer(name=layer_name)
    feature_extractor = keras.Model(inputs=pre_model.inputs, outputs=layer.output)

    def compute_loss(input_image, filter_index):
        activation = feature_extractor(input_image)
        # We avoid border artifacts by only involving non-border pixels in the loss.
        filter_activation = activation[:, 2:-2, 2:-2, filter_index]
        return tf.reduce_mean(filter_activation)


    @tf.function
    def gradient_ascent_step(img, filter_index, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = compute_loss(img, filter_index)
        # Compute gradients.
        grads = tape.gradient(loss, img)
        # Normalize gradients.
        grads = tf.math.l2_normalize(grads)
        img += learning_rate * grads
        return loss, img

    def initialize_image():
        # We start from a gray image with some random noise
        img = tf.random.uniform((1, img_width, img_height, 3))
        # ResNet50V2 expects inputs in the range [-1, +1].
        # Here we scale our random inputs to [-0.125, +0.125]
        return (img - 0.5) * 0.25


    def visualize_filter(filter_index):
        # We run gradient ascent for 20 steps
        iterations = 30
        learning_rate = 10.0
        img = initialize_image()
        for iteration in range(iterations):
            loss, img = gradient_ascent_step(img, filter_index, learning_rate)

        # Decode the resulting input image
        img = deprocess_image(img[0].numpy())
        return loss, img


    def deprocess_image(img):
        # Normalize array: center on 0., ensure variance is 0.15
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 0.15

        # Center crop
        img = img[25:-25, 25:-25, :]

        # Clip to [0, 1]
        img += 0.5
        img = np.clip(img, 0, 1)

        # Convert to RGB array
        img *= 255
        img = np.clip(img, 0, 255).astype("uint8")
        return img


    all_imgs = []
    for filter_index in range(32):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(filter_index)
        all_imgs.append(img)

    # Build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    n = 8
    m = 4
    cropped_width = img_width - 25 * 2
    cropped_height = img_height - 25 * 2
    width = n * cropped_width + (n - 1) * margin
    height = m * cropped_height + (m - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # Fill the picture with our saved filters

    for i in range(n):
        for j in range(m):
            img = all_imgs[i * m + j]
            stitched_filters[
                (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
                (cropped_height + margin) * j : (cropped_height + margin) * j
                + cropped_height,
                :,
            ] = img
    keras.preprocessing.image.save_img("stiched_filters.png", stitched_filters)

## End of createFilterImages


