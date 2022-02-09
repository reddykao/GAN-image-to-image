# micro_001.py
import os
import cv2
import numpy as np

A_path = 'H:/SSIM/microlab/image/0125mo/A/train'
B_path = 'H:/SSIM/microlab/image/0125mo/pinholeori_C/train'
AB_path ='H:/SSIM/microlab/image/0125mo/test_AC/train'

img_list_a = os.listdir(A_path)
num_imgs_a = len(img_list_a)
img_list_b = os.listdir(B_path)
num_imgs_b = len(img_list_b)

for n in range(num_imgs_a):
    name_A = img_list_a[n]
    path_A = os.path.join(A_path, name_A)
    #print(path_A)
    name_B = img_list_b[n]
    path_B = os.path.join(B_path, name_B)
    #print(path_B)
    name_AB = name_A
    path_AB = os.path.join(AB_path, name_AB)
    print(path_AB)

    im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_AB, im_AB)
#-------------------------------------
#END


