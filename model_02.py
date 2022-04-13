import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
import os
import numpy as np
import matplotlib.pyplot as plt


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]  #ajoute un axe
    return pred_mask[0]

def display(display_list):
    """
    :param display_list: une liste de 3 listes [input_list, target_list, predict_list]
    :return: affichage
    """
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    n_rows = len(display_list[0])
    n_cols = len(display_list)  # = 1, 2 ou 3
    for line in range(n_rows):
        for col in range(n_cols):
            plt.subplot(n_rows, n_cols, col+1 + n_cols*line)
            plt.title(title[col])
            plt.imshow( tf.keras.utils.array_to_img(tf.reshape(display_list[col][line],(5000,5000,1))))
            plt.axis('off')
    plt.show()

"""
def find_nearest(array, value):
    idx = np.argmin(np.abs(array - value))
    return idx
def apply_seuil(image, num_class):
    seuils = np.linspace(0, 1, num_class)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i,j] = find_nearest(seuils, image[i,j])
    return image
"""

def apply_new_seuil(image):
    res = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] >=0.6 and image[i,j] <0.75:
                res[i,j] = 1
            if image[i,j] >= 0.75:
                res[i,j] = 2
    return res

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices("GPU")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#num_class = 5
m = 10      #nombre d'exemples dans la data
percent = 0.8   #taille relative du training set

input_dir = './db3/topo/'
target_dir = './db3/dens_seuil/'

input_img_paths = [os.path.join(input_dir,filename)for filename in os.listdir(input_dir)][:m]
target_img_paths = [os.path.join(target_dir, filename) for filename in os.listdir(target_dir)][:m]

print("Number of images : ", len(input_img_paths))
print("The input images' paths are :", input_img_paths)
print("The target images' paths are:", target_img_paths)

#chargement des images Ã  partir des chemins
input_list = [np.load(path) for path in input_img_paths]
target_list = [np.load(path) for path in target_img_paths]

print(type(target_list[0]))
print(target_list[0].shape)
#padding pour une dimension puissance de deux:
#input_list = [np.reshape(np.pad(topo,3,'edge'),(256,256,1)) for topo in input_list]
#target_list = [np.reshape(np.pad(dens,3,'edge'),(256,256,1)) for dens in target_list]
#input_list = [np.reshape(np.pad(topo,3,'edge'),(5000,5000,1)) for topo in input_list]
#target_list = [np.reshape(np.pad(dens,3,'edge'),(5000,5000,1)) for dens in target_list]
#target_list = [apply_new_seuil(target_image) for target_image in target_list]
#target_list = [apply_seuil(target_image, num_class) for target_image in target_list]
#print("seuil appliqué")

#normalisation des inputs :
mean = np.mean(input_list)
std = np.std(input_list)
input_list = (input_list-mean)/std

#un tf.dataset Ã  partir de la data
input_filenames = tf.constant(input_list)
target_filenames = tf.constant(target_list)
full_dataset = tf.data.Dataset.from_tensor_slices((input_filenames, target_filenames))

#affichage des premiers Ã©lÃ©ments du dataset
num_disp = 5
images = [image for image, mask in full_dataset.take(num_disp)]
masks = [mask for image, mask in full_dataset.take(num_disp)]
display([images, masks])

train_size = int(percent * m)
training_set = full_dataset.take(train_size)
testing_set = full_dataset.skip(train_size)

def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Convolutional downsampling block

    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns:
        next_layer, skip_connection --  Next layer and skip connection outputs
    """

    ## the 2 convolutions:
    conv = Conv2D(n_filters,  # Number of filters
                  (3, 3),  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters,  # Number of filters
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)

    # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = MaxPooling2D()(conv)

    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection

def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block

    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns:
        conv -- Tensor output
    """

    up = Conv2DTranspose(
        n_filters,  # number of filters
        3,  # Kernel size
        strides=(2, 2),
        padding='same')(expansive_input)

    # Merge the previous output and the contractive_input

    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,  # Number of filters
                  (3, 3),  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters,  # Number of filters
                  (3, 3),  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)

    return conv

def unet_model(input_size=(256, 256, 1), n_filters=32,n_classes=3):
    """
    Unet model

    Arguments:
        input_size -- Input shape
        n_filters -- Number of filters for the convolutional layers
    Returns:
        model -- tf.keras.Model
    """
    inputs = Input(input_size)
    # Contracting Path (encoding)
    # Add a conv_block with the inputs of the unet_ model and n_filters
    filters = n_filters
    cblock1 = conv_block(inputs, n_filters)
    # Chain the first element of the output of each block to be the input of the next conv_block.
    # Double the number of filters at each new step
    cblock2 = conv_block(cblock1[0], 2 * filters)
    cblock3 = conv_block(cblock2[0], 4 * filters)
    cblock4 = conv_block(cblock3[0], 8 * filters, dropout_prob=0)  # Include a dropout_prob of 0.3 for this layer
    # Include a dropout_prob of 0.3 for this layer, and avoid the max_pooling layer
    cblock5 = conv_block(cblock4[0], 16 * filters, dropout_prob=0, max_pooling=False)

    # Expanding Path (decoding)
    # Add the first upsampling_block.
    ublock6 = upsampling_block(cblock5[0], cblock4[1], 8 * filters)
    # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
    # Note that you must use the second element of the contractive block i.e before the maxpooling layer.
    # At each step, use half the number of filters of the previous block
    ublock7 = upsampling_block(ublock6, cblock3[1], 4 * filters)
    ublock8 = upsampling_block(ublock7, cblock2[1], 2 * filters)
    ublock9 = upsampling_block(ublock8, cblock1[1], filters)

    conv9 = Conv2D(n_filters,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)

    # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
    conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1,activation = 'softmax', padding='same')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


unet = unet_model()
unet.summary()
"""
unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])"""

"""
unet.compile(optimizer=tf.keras.optimizers.SGD(),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])"""

unet.compile(optimizer=tf.keras.optimizers.Adamax(),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

EPOCHS = 300
VAL_SUBSPLITS = 5
BUFFER_SIZE = 500
BATCH_SIZE = 8
training_set_batched = training_set.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(training_set.element_spec)
model_history = unet.fit(training_set_batched,
                         epochs=EPOCHS,
                         use_multiprocessing='True')
#unet.save("saved_model/unet_classifier_07_04")


loss = model_history.history['loss']
plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()


#affichage des prÃ©dictions sur les 5 premiers Ã©lÃ©ments du training set
num_disp = 5
images = [image for image, mask in training_set.take(num_disp)]
masks = [mask for image, mask in training_set.take(num_disp)]
predictions = [create_mask(unet.predict(tf.reshape(image, (1,256,256,1)))) for image in images]

display([images,masks,predictions])

#affichage des prÃ©dictions sur les 5 derniers Ã©lÃ©ments du training set
num_disp = 5
images = [image for image, mask in training_set.skip(train_size - num_disp)]
masks = [mask for image, mask in training_set.skip(train_size - num_disp)]
predictions = [create_mask(unet.predict(tf.reshape(image, (1,256,256,1)))) for image in images]

display([images,masks,predictions])

#affichage des prédictions sur les 5 premiers éléments du training set
images = [image for image, mask in testing_set.take(num_disp)]
masks = [mask for image, mask in testing_set.take(num_disp)]
predictions = [create_mask(unet.predict(tf.reshape(image, (1,256,256,1)))) for image in images]
display([images,masks,predictions])
