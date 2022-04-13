import matplotlib.pyplot as plt
import numpy as np
import os

def display(display_list):
    """
    :param display_list: une liste de 3 listes [input_list, target_list, predict_list]
    :return: affichage
    """
    plt.figure(figsize=(15, 15))
    title = ['topo', 'vege brute', 'vege lissée + seuil']
    n_rows = len(display_list[0])
    n_cols = len(display_list)  # = 1, 2 ou 3
    for line in range(n_rows):
        for col in range(n_cols):
            plt.subplot(n_rows, n_cols, col+1 + n_cols*line)
            plt.title(title[col])
            plt.imshow(display_list[col][line])
            plt.axis('off')
    plt.show()


input_dir = './db512/topo/'
target_dir = './db512/dens/'
input_img_paths = [os.path.join(input_dir,filename)for filename in os.listdir(input_dir)][:5]
target_img_paths = [os.path.join(target_dir, filename) for filename in os.listdir(target_dir)][:5]
print("Number of images : ", len(target_img_paths))
print("The target images' paths are:", target_img_paths)
N = len(target_img_paths)
input_list = [np.load(path) for path in input_img_paths]
target_list = [np.load(path) for path in target_img_paths]


clean_dir = './db512/mask_50/'
clean_img_paths = [os.path.join(clean_dir, filename) for filename in os.listdir(clean_dir)][:5]
clean_list = [np.load(path) for path in clean_img_paths]
print(clean_dir)
print(clean_img_paths)
print(clean_list[0])
display([input_list, target_list, clean_list])

