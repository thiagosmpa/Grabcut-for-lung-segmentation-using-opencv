import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('chestxray.jpg')

mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

startXL = 40
startYL = 50
rect = (startXL, startYL, (320-startXL),(700-startYL))

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
leftlung = img*mask2[:,:,np.newaxis]

startXR = 340
startYR = 130
rect2 = (startXR, startYR, (590-startXR),(715-startYR))

cv2.grabCut(img,mask,rect2,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
rightlung = img*mask2[:,:,np.newaxis]

cv2.imshow('Original image', img)
cv2.imshow('Right lung', rightlung)
cv2.imshow('Left lung', leftlung)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.imshow(img)
# plt.colorbar()
# plt.show()