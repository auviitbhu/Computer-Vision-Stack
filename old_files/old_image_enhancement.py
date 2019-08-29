from __future__ import division
import numpy as np 
import cv2

# Its Output : https://drive.google.com/open?id=1AT7q4B5QiYxYJ5_Uy6lNyY92xPn_XFDp

cap = cv2.VideoCapture('output.mp4')
# cap.open('output.mp4')
# cap = cv2.VideoCapture(0)

def contrast_stretching(img):
	mini = 256
	maxi = -1
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			# pass
			if img.item(i, j) < mini:
				mini = img.item(i, j)
			if img.item(i, j) > maxi:
				maxi = img.item(i, j)

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			img[i][j] = ((img.item(i, j) - mini) / (maxi - mini))*255
			

def buoy1(image):
	b, g, r = cv2.split(image)
	gaussian = cv2.GaussianBlur(b,(19,19),3)
	laplacian = cv2.Laplacian(gaussian, cv2.CV_64F)
	#median=cv2.medianBlur(laplacian,3)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	laplacian = cv2.erode(laplacian, kernel, iterations = 1)
	laplacian = cv2.dilate(laplacian, kernel, iterations = 1)	
	laplacian = cv2.erode(laplacian, kernel, iterations = 1)
	laplacian = cv2.dilate(laplacian, kernel, iterations = 1)
	laplacian = cv2.erode(laplacian, kernel, iterations = 1)
	laplacian = cv2.dilate(laplacian, kernel, iterations = 1)
	laplacian = cv2.erode(laplacian, kernel, iterations = 1)
	laplacian = cv2.dilate(laplacian, kernel, iterations = 5)
	_,Mask_image = cv2.threshold(laplacian,0,255,cv2.THRESH_BINARY_INV) 
	#Mask_image=cv2.bitwise_not(Mask_image)
	
	#cv2.imshow("mid",laplacian)

	detector=cv2.SimpleBlobDetector()
	params = cv2.SimpleBlobDetector_Params()
	params.filterByArea=True
	params.filterByInertia = False
	params.filterByConvexity = False
	params.maxArea=2000
	detector=cv2.SimpleBlobDetector_create(params)
	cv2.imshow("mid",Mask_image)
	blobs=detector.detect(Mask_image)
	for blob in blobs:
		x=int(blob.pt[0])
		y=int(blob.pt[1])
		radius=int((blob.size)/2)
		cv2.circle(Mask_image, (x, y), radius, (0, 0, 255), -1)
	Median=cv2.medianBlur(Mask_image,5)
	cnts=cv2.findContours(Mask_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	contours = cnts[1]
	for contour in contours:
	  if(cv2.contourArea(contour)>100):
	  	approx = cv2.approxPolyDP(contour,3,True)
	  	x,y,w,h=cv2.boundingRect(approx)
	  	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

	# cv2.imshow('fads',Median)
	# cv2.waitKey(0)
	return image
	#return laplacian

def buoy(image, iii):
	#hsv_image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	#Mask_image=cv2.inRange(hsv_image,np.array([0,120,120]),np.array([255,255,255]))
	#image = iii
	b,g,r = cv2.split(image)
	#_,Mask_image = cv2.threshold(b,105,255,cv2.THRESH_BINARY)
	#_,Mask_image=cv2.threshold(b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	Mask_image = cv2.adaptiveThreshold(b,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,999,10)
	Mask_image=cv2.bitwise_not(Mask_image)
	median=cv2.medianBlur(Mask_image,5)
	kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
	dilation=cv2.dilate(median,kernel,iterations=1)	
	erosion=cv2.erode(dilation,kernel,iterations=1)
	cv2.imshow("mid",erosion)
	cv2.moveWindow("mid",680,0)
	detector=cv2.SimpleBlobDetector()
	params = cv2.SimpleBlobDetector_Params()
	params.filterByArea=True
	params.maxArea=1000
	detector=cv2.SimpleBlobDetector_create(params)
	blobs=detector.detect(erosion)
	for blob in blobs:
		x=int(blob.pt[0])
		y=int(blob.pt[1])
		radius=int((blob.size)/2)
		cv2.circle(erosion, (x, y), radius, (0, 0, 255), -1)
	Median=cv2.medianBlur(erosion,5)
	cnts=cv2.findContours(erosion, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	contours = cnts[1]
	cent_frame_y = int(iii.shape[0]/2)
	cent_frame_x = int(iii.shape[1]/2)
	#cv2.circle(iii, (cent_frame_x, cent_frame_y),2,(0,255,0),-1)
	for contour in contours:
	  if(cv2.contourArea(contour)>5000):
	  	approx = cv2.approxPolyDP(contour,3,True)
	  	x,y,w,h=cv2.boundingRect(approx)
	  	cv2.rectangle(iii,(x,y),(x+w,y+h),(0,255,0),2)
		cx = x+int(w/2)
		cy = y+int(h/2)
		#cv2.line(iii, (cent_frame_x, cent_frame_y), (cx, cy), (0,0,255),1)

	# cv2.imshow('fads',Median)
	# cv2.waitKey(0)
	return image,iii


ret, img = cap.read()
print ret

while ret: 
	
	if ret == False:
		continue
	img = cv2.resize(img, (680,420))
	blur = cv2.GaussianBlur(img,(5,5),0)
	blur = cv2.medianBlur(blur,5)
	bgr_planes = cv2.split(img)
	contrast_stretching(bgr_planes[0])
	contrast_stretching(bgr_planes[1])
	contrast_stretching(bgr_planes[2])
	clahe = cv2.createCLAHE(clipLimit=50.0,tileGridSize=(10,10)) 
	bgr_planes[0] = clahe.apply(bgr_planes[0])
	bgr_planes[1] = clahe.apply(bgr_planes[1])
	bgr_planes[2] = clahe.apply(bgr_planes[2])
	bgr = cv2.merge(bgr_planes)
	YCbCr = cv2.cvtColor(bgr,cv2.COLOR_BGR2YCR_CB)
	y, Cb, Cr = cv2.split(YCbCr)
	contrast_stretching(y)
	median = cv2.medianBlur(y,5)
	# laplacian = cv2.Laplacian(median,cv2.CV_32F) 
	YCbCr = cv2.merge((median, Cb, Cr))
	bgr = cv2.cvtColor(YCbCr, cv2.COLOR_YCR_CB2BGR)

	bgr, img = buoy(bgr, img)

	cv2.imshow("sd", img)
	cv2.moveWindow("sd",0,0)
	cv2.imshow("ji", bgr)
	cv2.moveWindow("ji",0,0)
	if (cv2.waitKey(1) & 0xff == 27):
		break

	ret ,img = cap.read()

cv2.destroyAllWindows()
