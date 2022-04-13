import matplotlib.pyplot as plt
import numpy as np
import os


def good_num(i):
    if i <10 :
        return "000"+ str(i)
    if i <100 :
        return "00"+ str(i)
    if i >=100 and i <1000:
        return "0" + str(i)
    if i>=1000 :
        return str(i)


target_dir = './db3/topo/'
target_img_paths = [os.path.join(target_dir, filename) for filename in os.listdir(target_dir)][:10]
print("Number of images : ", len(target_img_paths))
print("The target images' paths are:", target_img_paths)

#chargement des images à partir des chemins
target_list = [np.load(path) for path in target_img_paths]

print(target_img_paths)

for index, image in enumerate(target_list):
    new_target = image
    string = "./topo_good/topo"+good_num(index)+".npy"
    with open(string, "wb") as f:
        np.save(f, new_target)
