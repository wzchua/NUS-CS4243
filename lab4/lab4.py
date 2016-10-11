import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


sobelFilterChangeInX = np.array([[-1.0, 0.0, 1.0], 
                                [-2.0, 0.0, 2.0], 
                                [-1.0, 0.0, 1.0]])
sobelFilterChangeInY = np.array([[1.0, 2.0, 1.0], 
                                [0.0, 0.0, 0.0], 
                                [-1.0, -2.0, -1.0]])

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

def guass_kernels(size, sigma=1.0):
    ## returns a 2d guassian kernel
    if size<3:
        size = 3
    m = size/2
    x, y = np.mgrid[-m:m+1, -m:m+1]
    kernel = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    kernel_sum = kernel.sum()
    if not sum == 0:
        kernel = kernel/kernel_sum

    return kernel

#center: 2d array, windowHeight: integer 
def generateArrayIndexesForBorder(center, windowHeight, width, height):
    outputArray = []
    start = center - windowHeight/2

    #top row
    #outputArray.append(np.array(start))
    for i in range(windowHeight - 1):
        start[1] = (start[1] + 1)
        #print start
        if(start[1] < width):
            outputArray.append(np.array(start))
        else:
            break

    for j in range(windowHeight - 1):
        start[0] = start[0] + 1
        if(start[0] < height):
            outputArray.append(np.array(start))
        else:
            break

    for i in range(windowHeight - 1):
        start[1] = start[1] - 1
        if(start[1] >= 0):
            outputArray.append(np.array(start))
        else:
            break

    for j in range(windowHeight - 1):
        start[0] = start[0] - 1
        if(start[0] >= 0):
            outputArray.append(np.array(start))
        else:
            break
    #print center
    #print outputArray
    return outputArray

fileNames = ['checker', 'flower', 'test1', 'test2', 'test3']
stepsizes = [10, 1]
stepsizesStr = ['10', '1']
startingIdArray = [10, 0]
 
for b in range(2):
    for a in range(5):
        img = cv2.imread(fileNames[a] + '.jpg', cv2.IMREAD_GRAYSCALE)
        print img
        gx = MyConvolve(img, sobelFilterChangeInX)
        gy = MyConvolve(img, sobelFilterChangeInY)

        I_xx = gx * gx
        I_xy = gx * gy
        I_yy = gy * gy

        guassian_kernels_size3 = guass_kernels(3, 1)
        W_xx = MyConvolve(I_xx, guassian_kernels_size3)
        W_xy = MyConvolve(I_xy, guassian_kernels_size3)
        W_yy = MyConvolve(I_yy, guassian_kernels_size3)

        stepsize = stepsizes[b]
        startingId = startingIdArray[b]
        W_xx10 = W_xx[startingId::stepsize, startingId::stepsize]
        W_xy10 = W_xy[startingId::stepsize, startingId::stepsize]
        W_yy10 = W_yy[startingId::stepsize, startingId::stepsize]

        W = np.zeros((2, 2))
        responseM = np.zeros(W_xx10.shape)
        #print responseM.shape
        for i in range(W_xx10.shape[0]):
            for j in range(W_xx10.shape[1]):
                W[1, 1] = W_yy10[i, j]
                W[0, 1] = W_xy10[i, j]
                W[1, 0] = W_xy10[i, j]
                W[0, 0] = W_xx10[i, j]

                detW = np.linalg.det(W)
                traceW = np.trace(W)
                responseM[i, j] = detW - 0.06 * traceW * traceW

        max_response = responseM.max()
        print max_response
        response_thres = max_response * 0.1
        
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.hold(True)

        for i in range(responseM.shape[0]):
            for j in range(responseM.shape[1]):
                if(responseM[i, j] >= response_thres):
                    i10 = i * stepsizes[b] +stepsizes[b]
                    j10 = j * stepsizes[b] +stepsizes[b]
                    borderArray = generateArrayIndexesForBorder(np.array([j10,i10]), 9, img.shape[0], img.shape[1])
                    for k in range(len(borderArray)):
                        plt.scatter(borderArray[k][0], borderArray[k][1], marker='.', s=1, color='blue')
        #save img
        plt.savefig(fileNames[a] + '_corners_ss'+stepsizesStr[b]+'.png')
        plt.clf()
        plt.close()

