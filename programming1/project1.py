from socket import J1939_NLA_BYTES_ACKED
from PIL import Image
import os
import numpy as np


# Open images and convert filter2_img to b/w 
pwd = os.path.dirname(__file__)
img1 = Image.open(f'{pwd}/media/filter1_img.jpg')
img2 = Image.open(f'{pwd}/media/filter2_img.jpg')
img2 = img2.convert('L')

# Provided normalization constants and Gaussian filters
norm_3x3 = 1/16
gaus3x3 = np.array(
    [[1.0,2.0,1.0],
    [2.0,4.0,2.0],
    [1.0,2.0,1.0]])

deriv_XGaus3x3 = np.array(
    [[1.0,0.0,-1.0],
    [2.0,0.0,-2.0],
    [1.0,0.0,-1.0]])

deriv_YGaus3x3 = np.array(
    [[ 1.0, 2.0, 1.0],
     [ 0.0, 0.0, 0.0],
     [-1.0,-2.0,-1.0]])


norm_5x5 = 1/273
gaus5x5 = np.array(
    [[1.0,4.0,7.0,4.0,1.0],
    [4.0,16.0,26.0,16.0,4.0],
    [7.0,26.0,41.0,26.0,7.0],
    [4.0,16.0,26.0,16.0,4.0],
    [1.0,4.0,7.0,4.0,1.0]])



arr = np.array([
[2,0,3,4,1,7],
[15,12,2,1,0,5],
[14,11,1,0,1,3],
[12,10,0,0,0,1],
[11,8,1,0,0,4],
[7,5,1,2,3,6]])
f1 = np.array([[1,1,1],
      [0,0,0],
      [-1,-1,1]])

f2 = np.array([[1,0,0],[0,1,0],[0,0,1]])

# Function to convert the image argument to a zero padded numpy matrix (ndarry)
# and apply the Gaussian filter parameter (gmtrx), then return blurred image
def g_filter(image:Image, padding:int, norm:float, g_matrix:np.array):
    dim = g_matrix.shape[0]
    img_arr = np.pad(np.asarray(image, dtype=np.float64), padding)
    img_arr_new = img_arr.copy()
    dimx, dimy = img_arr.shape
    for x in range(img_arr.shape[0]-2):
        for y in range(img_arr.shape[1]-2):
            sub_matrix = img_arr[x:x+dim,y:y+dim]
            # Convolve
            prod = np.multiply(g_matrix, sub_matrix)
            prod = np.sum(prod)
            img_arr_new[x+dim//2][y+dim//2] = prod
    img_arr_new = np.clip(img_arr_new, 0, 255).astype(np.uint8)

    return img_arr_new+norm

# Returns the image resulting from a Sobel filter
def sobel_filter(orig_image:Image) -> Image:
    image1 = g_filter(orig_image, 1, 1.1, deriv_XGaus3x3)
    image2 = g_filter(orig_image, 1, 1.1, deriv_YGaus3x3)

    img1_arr = np.asarray(image1, dtype=np.float64).copy()
    img2_arr = np.asarray(image2, dtype=np.float64).copy()

    sobel_filtered =  np.sqrt(
        np.multiply(img1_arr, img1_arr)+np.multiply(img2_arr, img2_arr))

    return Image.fromarray(np.clip(sobel_filtered, 0, 255).astype(np.uint8), 'L')



#Uncomment as needed

## Original Images ##

#img1.show()
#img2.show()

## Display with initial Gaussian matrix & norm const for each image ##

print(g_filter(arr, 1, -1, f2))
#g_filter(img2, 2, norm_5x5, gaus5x5).show()

## Display derivatives of 3x3 Gaussian matrix for each image ##

#g_filter(img1, 1, norm_3x3, deriv_XGaus3x3).show()
#g_filter(img1, 1, norm_3x3, deriv_YGaus3x3).show()

## Display derivatives of 3x3 Gaussian matrix for each image ##

#g_filter(img2, 1, norm_5x5, deriv_XGaus3x3).show()
#g_filter(img2, 1, norm_5x5, deriv_YGaus3x3).show()

## Sobel filter ##

#sobel_filter(img1).show()
#sobel_filter(img2).show()





