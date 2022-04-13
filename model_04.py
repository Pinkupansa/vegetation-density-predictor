import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

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
            plt.imshow( tf.keras.utils.array_to_img(tf.reshape(display_list[col][line],(img_size,img_size,1))))
            plt.axis('off')
    plt.show()

def apply_new_seuil(image):
    res = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] >=0.6 and image[i,j] <0.75:
                res[i,j] = 1
            if image[i,j] >= 0.75:
                res[i,j] = 2
    return res

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        #ne pas afficher les messages de tensorflow
physical_devices = tf.config.list_physical_devices("GPU")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))  #afficher la liste des GPUs utilisés
tf.config.experimental.set_memory_growth(physical_devices[0], True)

m = 1      #nombre d'exemples dans la data
percent = 1   #taille relative du training set
img_size = 512

input_dir = './db512/topo/'
target_dir = './db512/mask_50/'

input_img_paths = [os.path.join(input_dir,filename)for filename in os.listdir(input_dir)][:m]
target_img_paths = [os.path.join(target_dir, filename) for filename in os.listdir(target_dir)][:m]

print("Number of images : ", len(input_img_paths))
print("The input images' paths are :", input_img_paths)
print("The target images' paths are:", target_img_paths)

#chargement des images Ã  partir des chemins
input_list = [np.load(path) for path in input_img_paths]
target_list = [np.load(path) for path in target_img_paths]
print("Images loaded")
print("Type of an input image : ",type(input_list[0]))
print("Shape of an input image :", input_list[0].shape)
print("Type of a target image : ",type(target_list[0]))
print("Shape of a target image :", target_list[0].shape)


#ajout d'une dimension
input_list = [np.reshape(topo,(img_size,img_size,1)) for topo in input_list]
target_list = [np.reshape(dens,(img_size,img_size,1)) for dens in target_list]


#normalisation des inputs :
maxi = 0
for image in input_list:
    maxi = max(maxi, np.amax(image))
input_list = [img / maxi for img in input_list]
print("Normalization faite ")

#un tf.dataset à partir de la data
input_filenames = tf.constant(input_list)
target_filenames = tf.constant(target_list)
full_dataset = tf.data.Dataset.from_tensor_slices((input_filenames, target_filenames))

#affichage des premiers Ã©lÃ©ments du dataset
"""num_disp = 5
images = [image for image, mask in full_dataset.take(num_disp)]
masks = [mask for image, mask in full_dataset.take(num_disp)]
display([images, masks])"""

#séparation en training et testing
train_size = int(percent * m)
training_set = full_dataset.take(train_size)
testing_set = full_dataset.skip(train_size)

print("the size of the training set is : ", train_size)


def conv_block(inputs=None, n_filters=32, dropout_prob=0.3, max_pooling=True):
    conv = Conv2D(n_filters,  # Number of filters
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)

    conv = BatchNormalization()(conv, training=False)
    conv = Conv2D(n_filters,  # Number of filters
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)

    conv = BatchNormalization()(conv, training=False)

    # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = MaxPooling2D(pool_size = (2,2))(conv)

    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection

def upsampling_block(expansive_input, contractive_input, n_filters=32):
    up = Conv2DTranspose(
        n_filters,  # number of filters
        (3,3),  # Kernel size
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

def unet_model(input_size=(img_size, img_size, 1), n_filters=32,n_classes=3):
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
    cblock4 = conv_block(cblock3[0], 8 * filters, dropout_prob=0.3)  # Include a dropout_prob of 0.3 for this layer
    # Include a dropout_prob of 0.3 for this layer, and avoid the max_pooling layer
    cblock5 = conv_block(cblock4[0], 16 * filters, dropout_prob=0.3, max_pooling=False)

    # Expanding Path (decoding)
    # Add the first upsampling_block.
    ublock6 = upsampling_block(cblock5[0], cblock4[1], 8 * filters)
    # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
    # Note that you must use the second element of the contractive block i.e before the maxpooling layer.
    # At each step, use half the number of filters of the previous block
    ublock7 = upsampling_block(ublock6, cblock3[1], 4 * filters)
    ublock8 = upsampling_block(ublock7, cblock2[1], 2 * filters)
    ublock9 = upsampling_block(ublock8, cblock1[1], filters)


    # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
    conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes,
                    1,
                    activation ='softmax',
                    padding='same',
                    kernel_initializer='he_normal')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


def dice_coef(y_true, y_pred, smooth=1):
    # flatten
    #y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # one-hot encoding y with 3 labels : 0=background, 1=label1, 2=label2
    y_true_f = tf.one_hot(K.cast(y_true, np.uint8), 3, dtype=tf.float32)
   # y_pred_f = tf.one_hot(K.cast(y_pred_f, np.uint8), 3, dtype=tf.float32)
    # calculate intersection and union exluding background using y[:,1:]
    intersection = K.sum(y_true_f[:,1:]* y_pred_f[:,1:], axis=[-1])
    union = K.sum(y_true_f[:,1:], axis=[-1]) + K.sum(y_pred_f[:,1:], axis=[-1])
    # apply dice formula
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)



"""
def diceloss(y_true, y_pred, smooth=1e-6):
    # flatten label and prediction tensors
    y_pred = K.flatten(y_pred)
    
    y_true = K.flatten(y_true)

    intersection = K.sum(K.dot(y_true, y_pred))
    dice = (2 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return 1 - dice
"""


def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2, 3)) + beta * K.sum(p1 * g0, (0, 1, 2, 3))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T

unet = unet_model()
unet.summary()

unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),
             loss=tversky_loss,
             metrics=['accuracy'])

EPOCHS = 100
BUFFER_SIZE = 500
BATCH_SIZE = 1

training_set_batched = training_set.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(training_set.element_spec)

model_history = unet.fit(training_set_batched,
                         epochs=EPOCHS,
                         use_multiprocessing='True')
#unet.save("saved_model/unet_classifier_13_04")


#affichage de la loss
loss = model_history.history['loss']
plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([-1, 1])
plt.legend()
plt.show()


#affichage des prÃ©dictions sur les 5 premiers Ã©lÃ©ments du training set
num_disp = 5
images = [image for image, mask in training_set.take(num_disp)]
masks = [mask for image, mask in training_set.take(num_disp)]
predictions = [create_mask(unet.predict(tf.reshape(image, (1,img_size, img_size,1)))) for image in images]

display([images,masks,predictions])


#affichage des prédictions sur les 5 premiers éléments du training set
images = [image for image, mask in testing_set.take(num_disp)]
masks = [mask for image, mask in testing_set.take(num_disp)]
predictions = [create_mask(unet.predict(tf.reshape(image, (1,img_size, img_size, 1)))) for image in images]
display([images,masks,predictions])
