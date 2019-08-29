import cv2
import numpy as np 

def homomorphic_filter(img_path):
    '''
    Input Image I(x,y)
    R(x,y) Reflectance, Data
    L(x,y) Illumination, noise signal(low frequency) 
    H(u,v) High pass filter

    # 1. Equation
    I(x,y) = R(x,y)*L(x,y)
    # 2. Take log
    ln(I(x,y)) = ln(R(x,y)) + ln(L(x,y))
    # 3. Frequency Domain (Fourier Transform)
    I'(u,v) = R'(u,v) + L'(u,v)
    # 4. Multiply H(u,v) both the sides
    I''(u,v) = ( R'(u,v) + L'(u,v) )*H(u,v)
    I''(u,v) = R''(u,v) + L''(u,v)
    # 5. Inverse Fourier Transform
    S(x,y) = P(x,y) + Q(x,y)
    # 6. Exponential
    exp(S(x,y)) = exp(P(x,y))*exp(Q(x,y))
    
    Note: Reflectance part vary Rapidly while illumination vary slowly by applying frequency domain
    filter we can reduce intensity variatiton across the image while highlightening the image data
    '''

    img = cv2.imread(img_path,-1)
	img = np.float32(img)
	img = img/255
	rows, cols, dim = img.shape

	#rh,rl are high frequency and low frequency gain respectively.the cutoff 32 is kept for 512,512 images
	#but it seems to work fine otherwise
    rh, rl, cutoff = 1.3,0.8,32

    #Blue, Green, Red channel
	b,g,r = cv2.split(img)

    # Step 2
	y_log_b = np.log(b+0.01)
	y_log_g = np.log(g+0.01)
	y_log_r = np.log(r+0.01)
	
    # Step 3. Frequency Domain
    y_fft_b = np.fft.fft2(y_log_b)
	y_fft_g = np.fft.fft2(y_log_g)
	y_fft_r = np.fft.fft2(y_log_r)
    y_fft_shift_b = np.fft.fftshift(y_fft_b)
	y_fft_shift_g = np.fft.fftshift(y_fft_g)
	y_fft_shift_r = np.fft.fftshift(y_fft_r)

    # Step 3 to 4 Construction of High pass filter
	#D0 is the cutoff frequency again a parameter to be chosen
	D0 = cols/cutoff
	H = np.ones((rows,cols))
	B = np.ones((rows,cols))
	for i in range(rows):
		for j in range(cols):
			H[i][j]=((rh-rl)*(1-np.exp(-((i-rows/2)**2+(j-cols/2)**2)/(2*D0**2))))+rl #DoG filter

    # Step 4. High Pass Filter
	result_filter_b = H * y_fft_shift_b
	result_filter_g = H * y_fft_shift_g
	result_filter_r = H * y_fft_shift_r

    # Step 5. Inverse Fourier Transform
	result_interm_b = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter_b)))
	result_interm_g = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter_g)))
	result_interm_r = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter_r)))

    # Step 6. Exponential
	result_b = np.exp(result_interm_b)
	result_g = np.exp(result_interm_g)
	result_r = np.exp(result_interm_r)

	result = np.zeros((rows,cols,dim))
	result[:,:,0] = result_b
	result[:,:,1] = result_g
	result[:,:,2] = result_r

	#norm_image = cv2.normalize(result,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	
	return(result)
