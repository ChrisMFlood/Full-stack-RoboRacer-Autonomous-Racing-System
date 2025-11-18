import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import yaml
import scipy
from scipy.ndimage import distance_transform_edt as edt
from PIL import Image
import os
import pandas as pd
import sys
import utils
import matplotlib.pyplot as plt

path = 'Stellenbosch_University_Electrical_and_Electronic_Engineering_Template__1_/Results/maps/'
map_name = 'Austin'
if os.path.exists(f"{path}{map_name}.png"):
    map_img_path = f"{path}{map_name}.png"
elif os.path.exists(f"{path}{map_name}.pgm"):
    map_img_path = f"{path}{map_name}.pgm"
else:
    raise Exception("Map not found!")

map_yaml_path = f"{path}{map_name}.yaml"
raw_map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
raw_map_img = raw_map_img.astype(np.float64)

# plt.figure()
# plt.imshow(raw_map_img, cmap='gray')
# plt.show()

def changeColour(img,val, white=True, bin=False):
    # binary image
    if bin:
        img[img <= 210.] = 0
        img[img > 210.] = 1
    DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    if white:
        print(np.argwhere(img == 1))
        start = np.argwhere(img == 1)[0]
        c=1
    else:
        start = np.argwhere(img == 0)[0]
        c=0

    print(f"Starting point: {start}")
    stack = [start]
    visited = set()
    while stack:
        current = stack.pop()
        if img[current[0], current[1]] != c:
            continue
        if (current[0], current[1]) in visited:
            continue
        for direction in DIRECTIONS:
            next_point = current + direction
            if next_point[0] < 0 or next_point[0] >= img.shape[0] or next_point[1] < 0 or next_point[1] >= img.shape[1]:
                continue
            if img[next_point[0], next_point[1]] == c:
                stack.append(next_point)
        img[current[0], current[1]] = val
        visited.add((current[0], current[1]))
    # current = start
    # for direction in DIRECTIONS:
    #     while True:
    #         next_point = current + direction
    #         if next_point[0] < 0 or next_point[0] >= img.shape[0] or next_point[1] < 0 or next_point[1] >= img.shape[1]:
    #             break
    #         if img[next_point[0], next_point[1]] == c:


plt.figure()
plt.imshow(raw_map_img)
plt.show()
changeColour(raw_map_img,10, white=True, bin=True)
plt.figure()
plt.imshow(raw_map_img)
plt.show()
changeColour(raw_map_img,20)
plt.figure()
plt.imshow(raw_map_img)
plt.show()
changeColour(raw_map_img,30)
plt.figure()
plt.imshow(raw_map_img)
plt.show()
changeColour(raw_map_img,40,False)
plt.figure()
plt.imshow(raw_map_img)
plt.show()
changeColour(raw_map_img,50,False)
plt.figure()
plt.imshow(raw_map_img)
plt.show()