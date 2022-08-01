import os
from PIL import Image
import numpy as np
import cv2

pwd = os.path.dirname(__file__)

# Already greyscale
frame1_a = Image.open(f'{pwd}/media/frame1_a.png')
frame1_b = Image.open(f'{pwd}/media/frame1_b.png')
frame2_a = Image.open(f'{pwd}/media/frame2_a.png')
frame2_b = Image.open(f'{pwd}/media/frame2_b.png')



dog_x = np.array(
    [[1.0,0.0,-1.0],
    [2.0,0.0,-2.0],
    [1.0,0.0,-1.0]])

dog_y = np.array(
    [[ 1.0, 2.0, 1.0],
     [ 0.0, 0.0, 0.0],
     [-1.0,-2.0,-1.0]])

# cv2
color = (0, 255, 0)
thickness = 1

# Function to convert the image argument to a zero padded numpy matrix (ndarry)
# and apply the Gaussian filter parameter (gmtrx), then return blurred image
def g_filter(image:Image, filt:np.array):
    norm = 1.1
    img_arr = np.pad(np.asarray(image, dtype=np.float32), 1)
    img_arr_new = img_arr.copy()

    for x in range(img_arr.shape[0]-2):
        for y in range(img_arr.shape[1]-2):
            sub_matrix = img_arr[x:x+3,y:y+3]
            # Convolve
            prod = np.multiply(filt, sub_matrix)
            prod = norm*np.sum(prod)
            img_arr_new[x+1][y+1] = prod
    img_arr_new = np.clip(img_arr_new, 0, 255)

    #Image.fromarray(img_arr_new, 'L').show()
    return img_arr_new


def optical_flow(frame1:Image, frame2:Image):
    cv2_frame1 = np.float32(cv2.cvtColor(np.asarray(frame1.convert('RGB')), cv2.COLOR_RGB2BGR))
    cv2_frame1_mag = np.float32(cv2_frame1)
    frame1 = np.pad(np.asarray(frame1, dtype=np.float32), 1)
    frame2 = np.pad(np.asarray(frame2, dtype=np.float32), 1)


    # DoG x,y filterd result for frame1
    dog_frame1_x = g_filter(frame1, dog_x)
    dog_frame1_y = g_filter(frame1, dog_y)

    # Ax = b
    # a_mtrx is A
    a_mtrx = np.zeros((9,2), np.float32)
    # Temporal Derivative
    temp_deriv = np.zeros((3,3), np.float32)

    # Slide the window over DoG_x and DoG_y result images simultaneously and
    # create OF equation
    for x in range(dog_frame1_x.shape[0]-4):
        for y in range(dog_frame1_x.shape[1]-4):
            sub_matrix_x = dog_frame1_x[x:(x+3),y:(y+3)]
            sub_matrix_y = dog_frame1_y[x:(x+3),y:(y+3)]

            subf1 = frame1[x:(x+3),y:(y+3)]
            subf2 = frame2[x:(x+3),y:(y+3)]
            temp_deriv= np.subtract(subf2, subf1)
            temp_deriv = np.reshape(temp_deriv, (9,1))

            a_mtrx[0:9, 0:1] = np.reshape(sub_matrix_x,(9,1)).copy()
            a_mtrx[0:9, 1:2] = np.reshape(sub_matrix_y,(9,1)).copy()

            # Solve equation
            # ata is (A^T)A, atb is (A^T)b, ata_pinv is inverted ata
            ata = np.asmatrix(np.matmul(np.transpose(a_mtrx), a_mtrx))
            ata_pinv = np.linalg.pinv(ata)
            ata_pinv = np.asarray(ata_pinv)
            atb = np.matmul(np.transpose(a_mtrx), temp_deriv) * -1

            v_xy = np.matmul(ata_pinv, atb)

            # Discard extremes
            v_xy[0,0] = 10 if v_xy[0,0] > 500 else v_xy[0,0]
            v_xy[1,0] = 10 if v_xy[1,0] > 500 else v_xy[1,0]
            v_xy[0,0] = 10 if v_xy[0,0] < 500 else v_xy[0,0]
            v_xy[1,0] = 10 if v_xy[1,0] < 500 else v_xy[1,0]

            l_2 = np.sqrt(pow(v_xy[1,0],2)+pow(v_xy[1,0],2))


            # Overlay results on frame
            cv2_frame1 = cv2.line(
                cv2_frame1,
                (y+1, x+1),
                (y+int(v_xy[0,0])+1, x+int(v_xy[1,0])+1),
                color,
                thickness) if (y % 20 == 0 and l_2 > 1) else cv2_frame1
            cv2_frame1_mag = cv2.circle(
                cv2_frame1,
                (y+1, x+1),
                0,
                color,
                thickness) if (l_2 > 1) else cv2_frame1_mag



    Image.fromarray(cv2_frame1.astype(np.uint8)).show()
    Image.fromarray(cv2_frame1_mag.astype(np.uint8)).show()


optical_flow(frame1_a, frame1_b)
optical_flow(frame2_a, frame2_b)
