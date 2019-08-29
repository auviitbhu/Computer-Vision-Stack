import cv2
import numpy as np

def blob_detector(img, original_img):
    b,g,r = cv2.split(img)

    #Mask of blue Channel
    mask_image = cv2.adaptiveThreshold(
            b,255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            999,10)
	mask_image = cv2.bitwise_not(mask_image)

    mask_image = cv2.medianBlur(mask_image,5)

    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
	dilation=cv2.dilate(mask_image,kernel,iterations=1)	
	erosion=cv2.erode(dilation,kernel,iterations=1)

    cv2.imshow('erosion', erosion)

    # Params Blob Detector
    params = cv2.SimpleBlobDetector_Params()
	params.filterByArea=True
	params.maxArea=1000
    detector=cv2.SimpleBlobDetector_create(params)
	
    # Detect Blob
    blobs = detector.detect(erosion)
    for blob in blobs:
		x=int(blob.pt[0])
		y=int(blob.pt[1])
		radius=int((blob.size)/2)
		cv2.circle(erosion, (x, y), radius, (0, 0, 255), -1)

    # Gate Detection
	count = cv2.findContours(erosion, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	contours = count[1]
	for contour in contours:
	  if(cv2.contourArea(contour)>5000):
	  	approx = cv2.approxPolyDP(contour,3,True)
	  	x,y,w,h = cv2.boundingRect(approx)
	  	cv2.rectangle(original_img,(x,y),(x+w,y+h),(0,255,0),2)
		cx = x+int(w/2)
		cy = y+int(h/2)

    
    return original_img
