import numpy as np
import cv2

def contrast_stretching(img):
    min_value = 257
    max_value = -1

    row, col = img.shape
    for i in range(row):
        for j in range(col):
            if img[i,j] < min_value:
                min_value = img[i,j]
            if img[i,j] > max_value:
                max_value = img[i,j]
    
    for i in range(row):
        for j in range(col):
            img[i,j] = ( (img[i,j] - min_value)/(max_value - min_value) )*255

def normalize(img):

    min_value = 257
    max_value = -1

    row, col = img.shape
    for i in range(row):
        for j in range(col):
            if img[i,j] < min_value:
                min_value = img[i,j]
            if img[i,j] > max_value:
                max_value = img[i,j]

    for i in range(row):
        for j in range(col):
            img[i,j] = ( img[i,j] - min_value )*( (255 - 0)/(max_value - min_value) + 0)
            
    return img