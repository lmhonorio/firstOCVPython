import numpy as np
import cv2
import matplotlib.pyplot as plt
import datetime
#import imutils
import time

class clsImage(object):
	"""Code Example using Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.
for 

Examples:
    obj = ImageTest.clsImage(<filename>)
    tuple = obj.imageShape

Attributes:
    imagemName:
    imagem:
    
Properties:
Authors:
    (LEO) Leonardo de Mello Honorio 
    (CHV) Carlos Henrique Valerio de Moraes
Last Modified:
    (LEO) : 04/01/2016

Version:
    v0.1


COMMONLY USED COMMANDS
		flow.shape	(480L, 640L, 2L)	tuple
		flow[...,0].shape	(480L, 640L)	tuple
		np.amax(flow[...,0])	2.9410453	float32
		np.amin(flow[...,0])	-2.5188873	float32

        #==========================================
        a = [2,3,4,5,6,7,8,9,0]
        xyz = [0,12,4,6,242,7,9]
        gen = (x for x in xyz if x not in a)
        for x in gen:
            print x
        #==========================================
    

"""
	imageName = ""
	imagem = ""

	#========================================
	def __init__(self, *args):       
		""" Constructor: stores the name and loads the image - error handler
			should be implemented in the calling method  """
		if len(args) == 0:
			pass
		else:
			self.imagemName = args[0]
			self.imagem = ""
			try:
				self.imagem = cv2.imread(imagemName)
				self.loaded = True;
			except:
				self.loaded = False;

	#========================================
	def TestGetMask(self,imagemName):
		img = cv2.imread(imagemName)
		pc = img[200:400:,200:400:] 
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray[200:400,200:400]=0
		cv2.imshow('image',gray)
		cv2.imshow('cut',pc)
		cv2.waitKey(0)

	#========================================
	def plotsometing(self):
		# evenly sampled time at 200ms intervals
		t = np.arange(0., 5., 0.2)
		# red dashes, blue squares and green triangles
		plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
		plt.show()

		cv2.waitKey(0)
		cv2.destroyAllWindows()

	#========================================
	def loadImage(self,nome):
		""" loads the image from file
			exemple:
				obj = ImageTest.clsImage()
				obj.loadImage(<filename>)
					"""
		try:
			self.imagem = cv2.imread(nome)
			self.loaded = True;
		except:
			self.loaded = False;


	#========================================
	def Print_GrayHistogram(self,nome):
		""" loads the image from file, converts to gray and prints its histogram
			dependences:
				matplotlib, cv2, numpy
			parameter:
				nome: string
			exemple:
				obj = ImageTest.clsImage()
				obj.Print_GrayHistogram(<filename>)
					"""
		try:
			imagem = cv2.imread(nome)
			gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
			hist = cv2.calcHist([gray],[0],None,[256],[0,256])


			plt.subplot(131), plt.hist(gray.ravel(),256,[0,256], normed = True)
			plt.title('Bitwise Histogram for gray scale picture')
			plt.subplot(132),plt.plot(hist)
			plt.title('Histogram for gray scale')
			plt.subplot(133),plt.imshow( cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB))
			plt.title('Histogram for gray scale')
			plt.show()
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			return 1
		except:
			return 0


	#========================================
	def imageShape(self,imag):
		return np.shape(imag)


	#=======================================
	def Convert2Gray(self,imag):
		""" convert a loaded image to gray
			exemple:
				obj = ImageTest.clsImage(<filename>)
				grayImg = obj.Convert2Gray()
					"""
		try:
			return cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
		except:
			return 0

	#=======================================
	def Convert2Gray(self):
		""" convert a loaded image to gray
			exemple:
				obj = ImageTest.clsImage(<filename>)
				grayImg = obj.Convert2Gray()
					"""
		if not self.loaded:
			return 0
 
		try:
			return cv2.cvtColor(self.imagem, cv2.COLOR_BGR2GRAY)
		except:
			return 0

	#=======================================
	def loadAndShow(self,filename):
		imagem = cv2.imread(filename)
		gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
		#hist = cv2.calcHist([gray],[0],None,[256],[0,256])
		cv2.imshow(filename,gray) 
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	#=======================================
	def CreateKernel(self,grid,div):
		return np.ones(grid,np.float32)/div


	#=======================================
	def ApplyBlur(self,imag,grid):
		return cv2.blur(imag,grid)


	#=======================================
	def ApplyLaplacian(self,imag):
		return cv2.Laplacian(imag,cv2.CV_64F)


	#=======================================
	def Applyfilter2D(self,imag,kernel):
		return cv2.filter2D(imag,-1,kernel)


	#=======================================
	@staticmethod
	def matchingTests(): 
		im_1 = cv2.imread('a8.jpg')
		im_2 = cv2.imread('a9.jpg')

		im_1 = cv2.resize(im_1,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
		im_2 = cv2.resize(im_2,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)

		im_1 = cv2.cvtColor(im_1, cv2.COLOR_BGR2GRAY)
		im_2 = cv2.cvtColor(im_2, cv2.COLOR_BGR2GRAY)
		#cv2.imshow('trilho',im_1)

		orb = cv2.ORB_create()
		fast = cv2.FastFeatureDetector_create()

		kp_1, des_1 = orb.detectAndCompute(im_1,None)
		kp_2, des_2 = orb.detectAndCompute(im_2,None)

		#
		#kp_1, des_1  = fast.detectAndCompute(im_1,None)
		#kp_2, des_2 =  fast.detectAndCompute(im_2,None)

		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		matches = bf.match(des_1,des_2)
		
		matches = sorted(matches, key = lambda x:x.distance)

		draw_params = dict(matchColor = (20,20,20), singlePointColor = (200,200,200),
							matchesMask = None,
							flags = 0)

		
		img3 = cv2.drawMatches(im_1,kp_1,im_2,kp_2,matches[0:50], None, **draw_params)
		ipt1 = matches[0].queryIdx
		pt1 = kp_1[ipt1]
		ip1 = (int(pt1.pt[0]), int(pt1.pt[1]))
		ipt2 = matches[0].trainIdx
		pt2 = kp_2[ipt2]
		ip2 = (int(pt2.pt[0]), int(pt2.pt[1]))

		delta = 10
		cut1 = im_1[ip1[1]-delta:ip1[1]+delta,ip1[0]-delta:ip1[0]+delta] 
		cut2 = cut1 * 0

		#calculo da entropia = baixa entropia significa pouca informacao para casar as imagens
		#areas com baixa entropia ou deverao ser ignoradas ou entram com pouco peso e os pixels
		#serao interpolados linearmente, mas sem peso no processo de homografia
		pp = np.cov(cut1)
		a,b = np.linalg.eig(pp)

		pp2 = np.cov(cut2)
		aa,bb = np.linalg.eig(pp2)
		
		plt.imshow(cv2.cvtColor(cut1,cv2.COLOR_GRAY2RGB))
		plt.show()

		#aqui visualiza os pontos capturados nas imagens
		#cv2.circle(im_1,(int(pt1.pt[0]), int(pt1.pt[1])),int(4),(250,50,250),4,2)
		#cv2.circle(im_2,(int(pt2.pt[0]), int(pt2.pt[1])),int(4),(250,50,250),4,2)


		#cv2.imshow('compilado',img3)
		cv2.imshow('pt1',im_1)
		cv2.imshow('pt2',im_2)
		plt.figure()
		plt.imshow(img3)
		plt.show()
		#cv2.imshow('trilho',im)

		#kp = fast.detect(im)
		#im2 = cv2.drawKeypoints(im_1, kp_1, im2, color=(255,0,0))
		#cv2.imshow('trilho22',im2)
		#cv2.imshow('trilho3',im2)
		cv2.waitKey(0)





	#=======================================
	def doSubPlot(self, plt, position, img, color_argument, title):
		try:
			plt.subplot(position),plt.imshow(cv2.cvtColor(img, color_argument)),plt.title(title)
			plt.xticks([]), plt.yticks([])
		except:
			pass
		return plt


	#===================================================================================
	def MotionEstimation(self):
		camera  = cv2.VideoCapture(0)
		firstFrame = None

	# loop over the frames of the video
		while True:
			# grab the current frame and initialize the occupied/unoccupied
			# text
			(grabbed, frame) = camera.read()
			text = "Unoccupied"

			# if the frame could not be grabbed, then we have reached the end
			# of the video

			if not grabbed:
				break

			# resize the frame, convert it to grayscale, and blur it
			frame = imutils.resize(frame, width=500)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (21, 21), 0)

			# if the first frame is None, initialize it
			if firstFrame is None:
				firstFrame = gray
				continue

			# compute the absolute difference between the current frame and
			# first frame
			frameDelta = cv2.absdiff(firstFrame, gray)
			thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

			# dilate the thresholded image to fill in holes, then find contours
			# on thresholded image
			thresh = cv2.dilate(thresh, None, iterations=2)
			image, contours, hierarchy =    cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			(_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

			# loop over the contours
			for c in cnts:
				# if the contour is too small, ignore it
				if cv2.contourArea(c) < 100:
					continue
				cv2.drawContours(frame, [c], 0, (0,255,0), 3)
				# compute the bounding box for the contour, draw it on the frame,
				# and update the text
				(x, y, w, h) = cv2.boundingRect(c)
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				text = "Occupied"



				# draw the text and timestamp on the frame
			cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
					(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

				# show the frame and record if the user presses a key
			cv2.imshow("Security Feed", frame)
			cv2.imshow("Thresh", thresh)
			cv2.imshow("Frame Delta", frameDelta)
			key = cv2.waitKey(1) & 0xFF

				# if the `q` key is pressed, break from the lop
			if key == ord("q"):
				break

		# cleanup the camera and close any open windows
		camera.release()
		cv2.destroyAllWindows()



    #===================================================================================
	def CaptureWebCam(self):
		cap = cv2.VideoCapture(0)

		while(True):
			# Capture frame-by-frame
			ret, frame = cap.read()

			# Our operations on the frame come here
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			ret,thresh = cv2.threshold(gray,127,255,0)
			(_,contours, _) = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			cv2.drawContours(gray,contours,-1,(50,255,50),-1)
			# Display the resulting frame
			cv2.imshow('frame',gray)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		# When everything done, release the capture
		cap.release()
		cv2.destroyAllWindows()



	#========================================
	def testSomeFunctions(self):
		grayImage = self.Convert2Gray()
		kernel = self.CreateKernel((5,5),25)
		blur2 = self.ApplyBlur(grayImage,(15,15))
		dst = self.Applyfilter2D(grayImage,kernel)

		laplacian = self.ApplyLaplacian(grayImage)
		laplacian2 = self.ApplyLaplacian(blur2)        

		plt.figure(1)
		self.doSubPlot(plt,221,self.imagem, cv2.COLOR_BGR2RGB, 'original')
		self.doSubPlot(plt,222,grayImage, cv2.COLOR_GRAY2RGB, 'gray')
		self.doSubPlot(plt,223,blur2, cv2.COLOR_GRAY2RGB, 'blur')
		self.doSubPlot(plt,224,dst, cv2.COLOR_GRAY2RGB, 'filter 2d')
		plt.show()

		cv2.waitKey(0)
		cv2.destroyAllWindows()


	#========================================
	def printVersions():
		print(cv2.__version__)
		print(np.__version__)