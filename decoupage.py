import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal
from PIL import Image

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

def good_num(i):
    if i <10 :
        return "000"+ str(i)
    if i <100 :
        return "00"+ str(i)
    if i >=100 and i <1000:
        return "0" + str(i)
    if i>=1000 :
        return str(i)

input_dir = './db3/topo/'
target_dir = './db3/dens/'
input_img_paths = [os.path.join(input_dir,filename)for filename in os.listdir(input_dir)]
target_img_paths = [os.path.join(target_dir, filename) for filename in os.listdir(target_dir)]

print("Number of images : ", len(target_img_paths))
print("The target images' paths are:", target_img_paths)
N = len(target_img_paths)
compteur = 0
print("N = ", N)

#input_list = [np.load(path) for path in input_img_paths]
#target_list = [np.load(path) for path in target_img_paths]

for index in range(N):
    print("on s'occupe de l'image ", index)
    big_topo = np.load(input_img_paths[index])
    big_dens = np.load(target_img_paths[index])
    print("img loaded")
    for i in range(9):
        for j in range(9):
            small_topo = big_topo[i*512 : (i+1)*512, j * 512: (j+1)*512]
            string = "./db512/topo/" + good_num(compteur) + ".npy"
            with open(string, "wb") as f:
                np.save(f, small_topo)
            small_dens = big_dens[i*512 : (i+1)*512, j * 512: (j+1)*512]
            string = "./db512/dens/" + good_num(compteur) + ".npy"
            with open(string, "wb") as f:
                np.save(f, small_dens)
            compteur = compteur + 1
            print(compteur)


"""
topo = input_list[0]
img = target_list[0]
img = Image.fromarray(img, mode = "L")
img.save("test01.jpg", quality=1)
img1 = Image.open("test01.jpg")
img1 = np.asarray(img1)
img = target_list[0]
print("applying conv")
kernel = np.ones((5, 5))
imgconvol = signal.convolve2d(img,
                              kernel,
                              mode='same')
imgconvol = imgconvol / 25.0
print("applying seuil")
img_finale = apply_new_seuil(imgconvol)
display([[topo], [img], [img_finale]])

print("applying conv")
kernel = np.ones((5, 5))
imgconvol = signal.convolve2d(img1,
                              kernel,
                              mode='same')
imgconvol = imgconvol / 25.0
print("applying seuil")
img_finale = apply_new_seuil(imgconvol)
display([[topo], [img1], [img_finale]])
"""

"""
target_list = [np.load(path) for path in target_img_paths]
kernel = np.ones((10,10))
for index, image in enumerate(target_list):
    imgconvol = signal.convolve2d(image,
                                  kernel,
                                  mode='same')
    imgconvol = imgconvol / 100.0
    img_finale = apply_new_seuil(imgconvol)
    string = "./mask_lisse/"+good_num(index)+".npy"
    with open(string, "wb") as f:
        np.save(f, img_finale)
"""