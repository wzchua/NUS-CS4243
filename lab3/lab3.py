import cv2
import numpy as np

#border handling is by extension
def MyConvolve(img, ff):
    result = np.zeros(img.shape)
    y = img.shape[0]
    x = img.shape[1]
 
    ff = np.fliplr(ff)
    ff = np.flipud(ff)
 
    fy = ff.shape[0]
    fx = ff.shape[1]
    fyMid = fy/2
    fxMid = fx/2
 
    for i in range(0, y):
        for j in range(0, x):
            sum = 0.0
            for ifY in range(fy):
                iy = i - fyMid + ifY #cap y within img y dimension
                if (iy < 0):
                    iy = 0
                elif(iy >= y):
                    iy = y - 1
 
                for jfX in range(fx):
                    jx = j - fxMid + jfX #cap x within img x dimension
                    if(jx < 0):
                        jx = 0
                    elif(jx >= x):
                        jx = x - 1
 
                    sum += ff[ifY, jfX] * img[iy, jx]
 
            result[i, j] = sum
 
    return result


def sqrtOfSumOfSquares(img1, img2):
	img1Sq = np.square(img1)
	img2Sq = np.square(img2)

	return np.sqrt(img1Sq + img2Sq)

def isMaximumAmongSides(img, i, j):
	center = img[i, j]

	if(i == 0):
		left = 0
	else:
		left = img[i-1, j]
	
	if(i == img.shape[0] - 1):
		right = 0
	else:
		right = img[i+1, j]
	isHorizontalStrongest = center > left and center > right

	if(j == 0):
		top = 0
	else:
		top = img[i, j-1]

	if(j == img.shape[1] - 1):
		bottom = 0
	else:
		bottom = img[i, j+1]

	isVerticalStrongest = center > top and center > bottom

	return isVerticalStrongest or isHorizontalStrongest


def nonMaximalSuppressionByNeighbour(img):
	x = img.shape[0]
	y = img.shape[1]

	for i in range(x):
		for j in range(y):
			if(not isMaximumAmongSides(img, i, j)):
				img[i, j] = 0

	return img

def normalize(img):
	maxValue = img.max()
	minValue = img.min()
	valueRange = maxValue - minValue
	img = (img - minValue) * (255.0 / valueRange)

	return img

guassianKernel = np.array([[1.0, 2.0, 1.0], 
							[2.0, 4.0, 2.0], 
							[1.0, 2.0, 1.0]])

sobelFilterChangeInX = np.array([[-1.0, 0.0, 1.0], 
								[-2.0, 0.0, 2.0], 
								[-1.0, 0.0, 1.0]])
sobelFilterChangeInY = np.array([[1.0, 2.0, 1.0], 
								[0.0, 0.0, 0.0], 
								[-1.0, -2.0, -1.0]])

prewitFilterChangeInX = np.array([[-1.0, 0.0, 1.0], 
								[-1.0, 0.0, 1.0], 
								[-1.0, 0.0, 1.0]])
prewitFilterChangeInY = np.array([[1.0, 1.0, 1.0], 
								[0.0, 0.0, 0.0], 
								[-1.0, -1.0, -1.0]])

fileNames = ['example', 'test1', 'test2', 'test3']

for a in range(4):
	img = cv2.imread(fileNames[a] + '.jpg', cv2.IMREAD_GRAYSCALE)
	imgFiltered = MyConvolve(img, guassianKernel)

	imgSobelChangeInX = MyConvolve(imgFiltered, sobelFilterChangeInX)
	imgSobelChangeInY = MyConvolve(imgFiltered, sobelFilterChangeInY)
	#imgSobelGradient = np.arctan2(imgSobelChangeInY, imgSobelChangeInX) * (180 / np.pi)

	imgPrewitChangeInX = MyConvolve(imgFiltered, prewitFilterChangeInX)
	imgPrewitChangeInY = MyConvolve(imgFiltered, prewitFilterChangeInY)
	#imgPrewitGradient = np.arctan2(imgPrewitChangeInY, imgPrewitChangeInX) * (180 / np.pi)

	imageSobel = sqrtOfSumOfSquares(imgSobelChangeInX, imgSobelChangeInY)
	imagePrewit = sqrtOfSumOfSquares(imgPrewitChangeInX, imgPrewitChangeInY)

	cv2.imwrite(fileNames[a] + '_sobel.jpg', normalize(imageSobel))
	cv2.imwrite(fileNames[a] + '_prewit.jpg', normalize(imagePrewit))

	thinnedImageSobel = nonMaximalSuppressionByNeighbour(imageSobel)
	cv2.imwrite(fileNames[a] + '_sobel_thinned.jpg', normalize(thinnedImageSobel))

	thinnedImagePrewit = nonMaximalSuppressionByNeighbour(imagePrewit)
	cv2.imwrite(fileNames[a] + '_prewit_thinned.jpg', normalize(thinnedImagePrewit))