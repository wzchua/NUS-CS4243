import cv2
import numpy as np

def bgrtohsv(bgr):
	b = bgr[0]
	g = bgr[1]
	r = bgr[2]

	rN = r/float(255)
	gN = g/float(255)
	bN = b/float(255)

	Cmax = max(rN, gN, bN)
	Cmin = min(rN, gN, bN)

	delta = Cmax - Cmin

	if delta == 0:
		hue = 0
	elif Cmax == rN:
		hue = (((gN - bN) / delta) % 6)
	elif Cmax == gN:
		hue = (((bN - rN) / delta) + 2)
	elif Cmax == bN:
		hue = (((rN - gN) / delta) + 4)

	if Cmax == 0:
		sat = 0
	else:
		sat = delta / Cmax

	v = Cmax

	hsv = np.zeros((3), dtype=np.uint8)
	hsv[0] = (hue * 255) / float(6)
	hsv[1] = sat * 255
	hsv[2] = v * 255

	#print hue * 60, sat, v

	return hsv

def hsvtobgr(hsv):
	h = hsv[0] * 360 / float(255)
	s = hsv[1] / float(255)
	v = hsv[2] / float(255)

	C = v * s
	X = C * (1 - abs(((h / 60) % 2) - 1))
	m = v - C

	if(h ==360) or (h < 60):
		rN = C
		gN = X
		bN = 0
	elif(h < 120):
		rN = X
		gN = C
		bN = 0
	elif(h < 180):
		rN = 0
		gN = C
		bN = X
	elif(h < 240):
		rN = 0
		gN = X
		bN = C
	elif(h < 300):
		rN = X
		gN = 0
		bN = C
	elif(h < 360):
		rN = C
		gN = 0
		bN = X

	bgr = np.zeros((3), dtype=np.uint8)

	bgr[2] = (rN + m) * 255
	bgr[1] = (gN + m) * 255
	bgr[0] = (bN + m) * 255

	return bgr


def BGRtoHSV(img):
	x = len(img[:,0,0])
	y = len(img[0,:,0])

	hsv = np.zeros((x, y, 3), dtype=np.uint8)

	for i in range(x):
		for j in range(y):		
			result = bgrtohsv(img[i,j,:])

			hsv[i, j, :] = result

	return hsv

def HSVtoBGR(img):
	x = len(img[:,0, 0])
	y = len(img[0,:, 0])
	
	BGR = np.zeros((x, y, 3), dtype=np.uint8)

	for i in range(x):
		for j in range(y):		
			bgr = hsvtobgr(img[i, j, :])
			BGR[i, j,:] = bgr

	return BGR

def equalizationHSV(img):
	v = img[:, :, 2]
	freq = np.zeros(256)
	x = len(img[:,0, 0])
	y = len(img[0,:, 0])
	n = x * y
	bin = n / 256

	for i in range(x):
		for j in range(y):		
			freq[v[i, j]] +=1

	culFreq = np.zeros(256)
	
	culFreq[0] = freq[0]

	for i in range(1, 256):
		culFreq[i] = freq[i]
		culFreq[i] += culFreq[i-1]

	h = np.zeros(256, dtype=np.uint8)
	h[i] = 0;

	for i in range(1, 256):
		h[i] = culFreq[i] / float(n) * 255

	for i in range(x):
		for j in range(y):
			v[i, j] = h[v[i,j]]

	img[:, :, 2] = v

	print freq
	print culFreq

	print h

	return img

fileNames = ['concert', 'sea1', 'sea2']


for a in range(1):
	img = cv2.imread(fileNames[a]+'.jpg')

	print img[:,:,0]
	hsvImg = BGRtoHSV(img)
	#print hsvImg[:,:,0]
	#print hsvImg[:,:,1]
	#print hsvImg[:,:,2]

	cv2.imwrite(fileNames[a]+'_hue.jpg', hsvImg[:,:,0])
	cv2.imwrite(fileNames[a]+'_saturation.jpg', hsvImg[:,:,1])
	cv2.imwrite(fileNames[a]+'_brightness.jpg', hsvImg[:,:,2])

	hImg = cv2.imread(fileNames[a]+'_hue.jpg', -1)
	sImg = cv2.imread(fileNames[a]+'_saturation.jpg', -1)
	vImg = cv2.imread(fileNames[a]+'_brightness.jpg', -1)

	hsvImg2 = np.dstack((hImg, sImg, vImg))
	print 'read'
	#print hImg
	#print sImg
	#print vImg

	bgrImg = HSVtoBGR(hsvImg2)
	print bgrImg[:,:,0]

	cv2.imwrite(fileNames[a] + '_hsv2rgb.jpg', bgrImg)

	eHSVImg2 = equalizationHSV(hsvImg2)

	eBGRImg = HSVtoBGR(eHSVImg2)

	cv2.imwrite(fileNames[a] + '_histeq.jpg', eBGRImg)

