#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 00:38:46 2022

@author: jyunpingkao
"""
import cv2 as cv
from skimage.metrics import structural_similarity as ssim

source_path= '71.png'
image1 = cv.imread(source_path)
image1 = cv.cvtColor(image1,cv.COLOR_BGR2GRAY)

generate_path= 'a9.png'
image2 = cv.imread(generate_path)
image2 = cv.cvtColor(image2,cv.COLOR_BGR2GRAY) 

ssim_value = ssim(image1, image2)

print("ssim value:" )
print(ssim_value)
