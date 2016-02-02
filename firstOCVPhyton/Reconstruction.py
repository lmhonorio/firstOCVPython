import cv2
import numpy as np
import matplotlib.pyplot as plt



class clsReconstruction(object):
	"""description of class"""


	@staticmethod
	def matchingTests(): 
		im_1 = cv2.imread('c1.bmp')
		im_2 = cv2.imread('c2.bmp')

		k = ([[ 683.39404297,    0.        ,  267.21336591], [   0.        ,  684.3449707 ,  218.56421036],  [   0.        ,    0.        ,    1.        ]])

		#resise, if it is necessary
		#im_1 = cv2.resize(im_1,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
		#im_2 = cv2.resize(im_2,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)

		#convert to gray
		im_1 = cv2.cvtColor(im_1, cv2.COLOR_BGR2GRAY)
		im_2 = cv2.cvtColor(im_2, cv2.COLOR_BGR2GRAY)


		#proceed with sparce feature matching
		orb = cv2.ORB_create()
		
		kp_1, des_1 = orb.detectAndCompute(im_1,None)
		kp_2, des_2 = orb.detectAndCompute(im_2,None)

		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		matches = bf.match(des_1,des_2)
		
		matches = sorted(matches, key = lambda x:x.distance)

		draw_params = dict(matchColor = (20,20,20), singlePointColor = (200,200,200),
							matchesMask = None,
							flags = 0)

		
		im_3 = cv2.drawMatches(im_1,kp_1,im_2,kp_2,matches[0:20], None, **draw_params)
		
		
		#select points to evaluate the fundamental matrix
		pts1 = []
		pts2 = []
		idx =  matches[1:20]

		for i in idx:
			pts1.append(kp_1[i.queryIdx].pt)
			pts2.append(kp_2[i.trainIdx].pt)
	

		
		pts1 = np.array(pts1)
		pts2 = np.array(pts2)

		#creating homegeneous coordenate
		pones = np.ones((1,len(pts1))).T

		pth_1 = np.hstack((pts1,pones))
		pth_2 = np.hstack((pts2,pones))

		k = np.array(k)
		ki = np.linalg.inv(k)
		#normalize the points
		ptn_1 = []
		ptn_2 = []
		for i in range(0,len(pts1)):
			ptn_1.append(ki * pth_1[i])
			ptn_2.append(ki * pth_2[i])


		#evaluate the fundamental Matrix
		#E, mask0 = cv2.findEssentialMat(pts1,pts1,k,cv2.FM_RANSAC)
		F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)	


		#selecting only inlier points
		pts1 = pts1[mask.ravel() == 1]
		pts2 = pts2[mask.ravel() == 1]

		# Find epilines corresponding to points in right image (second image) and
		# drawing its lines on left image
		lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
		lines1 = lines1.reshape(-1,3)
		img5,img6 = clsReconstruction.drawlines(im_1,im_2,lines1,pts1,pts2)
		# Find epilines corresponding to points in left image (first image) and
		# drawing its lines on right image
		lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
		lines2 = lines2.reshape(-1,3)
		img3,img4 = clsReconstruction.drawlines(im_2,im_1,lines2,pts2,pts1)
		plt.subplot(131),plt.imshow(img5)
		plt.subplot(132),plt.imshow(img3)
		plt.subplot(133),plt.imshow(im_3)

		plt.show()

			
	
	@staticmethod	
	def drawlines(img1,img2,lines,pts1,pts2):
		''' img1 - image on which we draw the epilines for the points in img2
		lines - corresponding epilines '''
		r,c = img1.shape
		img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
		img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
		for r,pt1,pt2 in zip(lines,pts1,pts2):
			color = tuple(np.random.randint(0,255,3).tolist())
			x0,y0 = map(int, [0, -r[2]/r[1] ])
			x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
			img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
			img1 = cv2.circle(img1,tuple((int(pt1[0]),int(pt1[1]))),5,color,-1)
			img2 = cv2.circle(img2,tuple((int(pt2[0]),int(pt2[1]))),5,color,-1)
		return img1,img2		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
			
		##for dense matching = not using right now
		#ipt1 = matches[0].queryIdx
		#pt1 = kp_1[ipt1]
		#ip1 = (int(pt1.pt[0]), int(pt1.pt[1]))
		#ipt2 = matches[0].trainIdx
		#pt2 = kp_2[ipt2]
		#ip2 = (int(pt2.pt[0]), int(pt2.pt[1]))

		#delta = 10
		#cut1 = im_1[ip1[1]-delta:ip1[1]+delta,ip1[0]-delta:ip1[0]+delta] 
		#cut2 = cut1 * 0

		##calculo da entropia = baixa entropia significa pouca informacao para casar as imagens
		##areas com baixa entropia ou deverao ser ignoradas ou entram com pouco peso e os pixels
		##serao interpolados linearmente, mas sem peso no processo de homografia
		#pp = np.cov(cut1)
		#a,b = np.linalg.eig(pp)

		#pp2 = np.cov(cut2)
		#aa,bb = np.linalg.eig(pp2)
		
		#plt.imshow(cv2.cvtColor(cut1,cv2.COLOR_GRAY2RGB))
		#plt.show()

		##aqui visualiza os pontos capturados nas imagens
		##cv2.circle(im_1,(int(pt1.pt[0]), int(pt1.pt[1])),int(4),(250,50,250),4,2)
		##cv2.circle(im_2,(int(pt2.pt[0]), int(pt2.pt[1])),int(4),(250,50,250),4,2)


		##cv2.imshow('compilado',img3)
		#plt.figure()
		#plt.imshow(img3)
		#plt.show()
		##cv2.imshow('trilho',im)

		##kp = fast.detect(im)
		##im2 = cv2.drawKeypoints(im_1, kp_1, im2, color=(255,0,0))
		##cv2.imshow('trilho22',im2)
		##cv2.imshow('trilho3',im2)
		#cv2.waitKey(0)



