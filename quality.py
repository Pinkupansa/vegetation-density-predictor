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

def apply_new_seuil(image):
    res = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] >=0.6 and image[i,j] <0.75:
                res[i,j] = 1
            if image[i,j] >= 0.75:
                res[i,j] = 2
            else :
                image[i,j] = 0
    return res


input_dir = './db512/topo/'
target_dir = './db512/dens/'
input_img_paths = [os.path.join(input_dir,filename)for filename in os.listdir(input_dir)]
target_img_paths = [os.path.join(target_dir, filename) for filename in os.listdir(target_dir)][0:1000]

print("Number of images : ", len(target_img_paths))
print("The target images' paths are:", target_img_paths)

#input_list = [np.load(path) for path in input_img_paths]
#target_list = [np.load(path) for path in target_img_paths]

N = len(target_img_paths)
kernel = np.ones((50,50))
compteur = 0
for i in range(N):
    image = np.load(target_img_paths[i])
    imgconvol = signal.convolve2d(image,
                                  kernel,
                                  mode='same')
    imgconvol = imgconvol / 2500.0
    img_finale = apply_new_seuil(imgconvol)
    string = "./db512/mask_50/"+good_num(compteur)+".npy"
    with open(string, "wb") as f:
        np.save(f, img_finale)
    print(compteur)
    compteur +=1
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