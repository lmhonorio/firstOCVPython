import numpy as np
import cv2
import matplotlib.pyplot as plt
import ImageTest
import HaarCascade
import CameraCalibration
import Reconstruction

file = '.\haarfiles\haarcascade_frontalface_alt.xml'


#CameraCalibration.clsCameraCalibration.GenerateImageDataset('a_pic',20,5,5)
#CameraCalibration.clsCameraCalibration.CalibrateUsingImages()

#obj = ImageTest.clsImage('.\images\a8.jpg')



#ImageTest.clsImage.matchingTests()


Reconstruction.clsReconstruction.matchingTests();

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






