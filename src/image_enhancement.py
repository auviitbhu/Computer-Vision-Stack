import numpy as np
import cv2

from filters import contrast_stretching

def enhance_img(img):
    b,g,r = cv2.split(img)

    contrast_stretching(b)
    contrast_stretching(g)
    contrast_stretching(r)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=50.0,tileGridSize=(10,10)) 
	b = clahe.apply(b)
	g = clahe.apply(g)
	r = clahe.apply(r)

	img = cv2.merge(b,g,r)
    
	YCbCr = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
	y, Cb, Cr = cv2.split(YCbCr)
	contrast_stretching(y)
    median = cv2.medianBlur(y,5)
	YCbCr = cv2.merge((median, Cb, Cr))

	img = cv2.cvtColor(YCbCr, cv2.COLOR_YCR_CB2BGR)

    return img