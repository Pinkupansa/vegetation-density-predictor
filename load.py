import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def display(display_list):
    """
    :param display_list: une liste de 3 listes [input_list, target_list, predict_list]
    :return: affichage
    """
    plt.figure(figsize=(10, 10))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    n_rows = len(display_list[0])
    n_cols = len(display_list)  # = 1, 2 ou 3
    for line in range(n_rows):
        for col in range(n_cols):
            plt.subplot(n_rows, n_cols, col+1 + n_cols*line)
            plt.title(title[col])
            plt.imshow( tf.keras.utils.array_to_img(tf.reshape(display_list[col][line],(256,256,1))))
            plt.axis('off')
    plt.show()

model = keras.models.load_model("saved_model/unet_classifier02")

m = 2
target_dir = './Topo/'
topos = sorted([os.path.join(target_dir, filename) for filename in os.listdir(target_dir)])[0:m]
print("Number of images : ", len(topos))
topos = [np.load(path) for path in topos]
topos = [np.reshape(np.pad(dens,3,'edge'),(256,256,1)) for dens in topos]

predictions = [create_mask(model.predict(tf.reshape(topo, (1,256,256,1)))) for topo in topos]
display([topos,predictions])

