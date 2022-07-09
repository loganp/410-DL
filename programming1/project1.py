from PIL import Image
import os
import numpy as np


def addPadding(image:Image, padding:int):
    image = Image.new(
        image.mode, 
        (image.height+padding*2, image.width+padding*2), 0).paste(
            image, (padding,padding))




def g_filter_pixel(n:np.ndarray, g_matrix:np.ndarray, sub_matrix:np.ndarray):
    new_val = 0
    for x in range(g_matrix.height):
        for y in range(g_matrix.height):
            new_val += n*g_matrix[x][y]*sub_matrix[x][y]
    
    sub_matrix[sub_matrix.height//2][sub_matrix.height//2] = new_val


def g_filter(image:Image, padding:int, norm:float, gmtrx:np.ndarray):
    addPadding(image, padding)
    img_arr = np.asarray(image)

    for x in range(img_arr.shape[0]-(padding // 2 + 1)):

        for y in range(img_arr.shape[1]-(padding // 2 + 1)):
            
            sub_array = img_arr[x:x+gmtrx.size[0],y:y+gmtrx.size[1]]
            g_filter_pixel(norm, gmtrx, sub_array)

    return np.fromarray(img_arr)

pwd = os.path.dirname(__file__)
img1 = Image.open(f'{pwd}/media/filter1_img.jpg')
img2 = Image.open(f'{pwd}/media/filter2_img.jpg')



norm3x3 = 1/16
mtrx3x3 = np.array(
    [[1,2,1],
    [2,4,2],
    [1,2,1]])


norm5x5 = 1/273
mtrx5x5 = np.array(
    [[1,4,7,4,1],
    [4,16,26,16,4],
    [7,26,41,26,7],
    [4,16,26,16,4],
    [1,4,7,4,1]])



image  = g_filter(img1, 1, norm3x3, mtrx3x3)



#print(np.asarray(img1))


