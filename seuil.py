import matplotlib.pyplot as plt
import numpy as np
import os

#Trouve l'indice de l'élement de array le plus proche de value
def find_nearest(array, value):
    idx = np.argmin(np.abs(array - value))
    return idx

def apply_seuil(image, num_class):
    seuils = np.linspace(0, 1, num_class)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i,j] = find_nearest(seuils, image[i,j])
    #print("done")
    return image

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

def good_num(i):
    if i <10 :
        return "000"+ str(i)
    if i <100 :
        return "00"+ str(i)
    if i >=100 and i <1000:
        return "0" + str(i)
    if i>=1000 :
        return str(i)

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(display_list[i])
    plt.axis('off')
  plt.show()


target_dir = './db3/dens/'
target_img_paths = [os.path.join(target_dir, filename) for filename in os.listdir(target_dir)]
print("Number of images : ", len(target_img_paths))
print("The target images' paths are:", target_img_paths)

#chargement des images à partir des chemins
target_list = [np.load(path) for path in target_img_paths]

for index, image in enumerate(target_list):
    new_target = apply_new_seuil(image)
    string = "./db3/dens_seuil/"+good_num(index)+".npy"
    with open(string, "wb") as f:
        np.save(f, new_target)
