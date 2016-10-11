import cv2
import numpy as np

cap = cv2.VideoCapture('traffic.mp4')

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

print 'width : ', width
print 'height : ', height
print 'fps : ', fps
print 'frame count : ', frame_count

widthInt = int(width)
heightInt = int(height)
fpsInt = int(fps)
frame_countInt = int (frame_count)

_,img = cap.read()
avgImg = np.float32(img)

for fr in range(1, frame_countInt):
	_,img = cap.read()
	alpha = 1.0/float(fr + 1)
	cv2.accumulateWeighted(img, avgImg, alpha)
	normImg = cv2.convertScaleAbs(avgImg)

	cv2.imshow('img', img)
	cv2.imshow('normImg', normImg)
	print "fr = ", fr, "alpha = ", alpha

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()

cv2.imwrite('background.jpg', normImg)