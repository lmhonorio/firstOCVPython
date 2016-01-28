import numpy as np
import cv2
import matplotlib.pyplot as plt
import ImageTest
import HaarCascade


file = '.\haarfiles\haarcascade_frontalface_alt.xml'


#obj = ImageTest.clsImage('.\images\a8.jpg')


im_1 = cv2.imread('trilhoa9.png')
im_2 = cv2.imread('a9.jpg')

cv2.imshow('trilho',im_1)

orb = cv2.ORB_create()
fast = cv2.FastFeatureDetector_create()

kp_1, des_1 = orb.detectAndCompute(im_1,None)
kp_2, des_2 = orb.detectAndCompute(im_2,None)

#kp_1, des_1  = fast.detectAndCompute(im_1,None)
#kp_2, des_2 =  fast.detectAndCompute(im_2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des_1,des_2)

matches = sorted(matches, key = lambda x:x.distance)

draw_params = dict(matchColor = (200,200,200), singlePointColor = (200,200,200),
                   matchesMask = None,
                   flags = 0)


img3 = cv2.drawMatches(im_1,kp_1,im_2,kp_2,matches[0:50], None, **draw_params)

plt.imshow(img3)
plt.show()
#cv2.imshow('trilho',im)

#kp = fast.detect(im)
im2 = cv2.drawKeypoints(im_1, kp_1, im2, color=(255,0,0))
cv2.imshow('trilho22',im2)
#cv2.imshow('trilho3',im2)
cv2.waitKey(0)

#obj.MotionEstimation()
#obj.CaptureWebCam()
#obj.Print_GrayHistogram('.\images\lena.jpg')


#List = list()
#List.append(HaarCascade.clsObjectConfidence(1,2))

#List = list(HaarCascade.clsObjectConfidence)

#objTest = ImageTest.clsImage('.\images\maca.jpg')
#objTest.OpticalFlow()
#objTest.OpticalFlow2()
#objTest.testSomeFunctions()


#Haar = HaarCascade.clsHaarCascade()
#Haar.OpticalFlow()
#Haar.runFaceIdentification(file, 50, 40, 2)

print('everything working fine!!')






