import numpy as np
import cv2
import glob
import time



class clsCameraCalibration(object):
	"""description of class"""

	def __init__(self):
		pass

	@staticmethod
	def CalibrateUsingImages():
		# termination criteria
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		ix = 9
		iy = 6
		# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		objp = np.zeros((iy*ix,3), np.float32)
		objp[:,:2] = np.mgrid[0:ix,0:iy].T.reshape(-1,2)

		# Arrays to store object points and image points from all the images.
		objpoints = [] # 3d point in real world space
		imgpoints = [] # 2d points in image plane.

		images = glob.glob('apic*.jpg')

		im = cv2.imread('calib_radial.jpg')
		gr = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		re, corn = cv2.findChessboardCorners(gr, (7,6),None) 

		i = 0
		for fname in images:
			img = cv2.imread(fname)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

			# Find the chess board corners
			ret, corners = cv2.findChessboardCorners(gray, (ix,iy),None)

			# If found, add object points, image points (after refining them)
			if ret == True:
				objpoints.append(objp)
				i += 1
				corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
				imgpoints.append(corners2)

				# Draw and display the corners
				img = cv2.drawChessboardCorners(img, (ix,iy), corners2,ret)
				cv2.imshow('img',img)
				cv2.waitKey(500)

		cv2.destroyAllWindows()

		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

		img = cv2.imread(images[0])
		h,  w = img.shape[:2]
		newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

		# undistort
		dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

		# crop the image
		x,y,w,h = roi
		dst = dst[y:y+h, x:x+w]
		cv2.imwrite('calibresult.png',dst)

		tot_error = 0

		mean_error = 0
		for i in range(len(objpoints)):
			imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
			error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
			tot_error += error

		print('total error: ', mean_error/len(objpoints))
		print (newcameramtx)
		print (repr(newcameramtx)) 

		f = open('matrix_k.py', 'w')
		f.writelines(repr(newcameramtx))
		f.close()

	@staticmethod
	def GenerateImageDataset(tagname, delay_time, time_interval, nsamples):
		
		video_capture = cv2.VideoCapture(0)
		font = cv2.FONT_HERSHEY_SIMPLEX
		t0 = time.time()
		tini = time.time()
		i = 0

		while True:
			# Capture frame-by-frame
			tnow = time.time()
			t1 = time.time()


			rec, frame = video_capture.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			cv2.putText(frame,'timer: ' + str(delay_time-(tnow-tini)) ,(5,25), font, 0.5,(255,255,25),2) 
			cv2.imshow('Video', frame)


			if tnow - tini > delay_time:
				if t1 - t0 > time_interval:
					t0 = time.time()
					i += 1
					cv2.imwrite(tagname + str(i) + '.jpg',gray)
					cv2.imshow(tagname + str(i), gray)


			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			if i >= nsamples:
				break

		# When everything is done, release the capture
		video_capture.release()
		cv2.destroyAllWindows()




