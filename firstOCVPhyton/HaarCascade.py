import numpy as np
import math
import cv2 
from matplotlib import pyplot as plt
from MultipleObjectEstimation import clsObjectConfidence
from MultipleObjectEstimation import clsMultipleObjectEstimation
                 

class clsHaarCascade(object):
    """description of class"""



#===================================================================================
    def CaptureWebCam(self):
        cap = cv2.VideoCapture(0)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv2.imshow('frame',gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

#===================================================================================
    def createCascade(self, cascfile):

        try:
            self.faceCascade = cv2.CascadeClassifier(cascfile)
            print('facecascade is loaded')
        except:
            print('error while loading face cascade')


    #===================================================================================
    def IdentifyGoodFeatures(self,old_frame, blur, mask):
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 500,
                                qualityLevel = 0.05,
                                minDistance = 5,
                                blockSize = 3 )

        old_frame = cv2.GaussianBlur(old_frame, blur, 0)
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask= None , **feature_params)


        return old_frame, old_gray, p0

    #===================================================================================
    def OpticalFlow(self):

            #fixed parameters
            cap = cv2.VideoCapture(0)
            blur = (25,25)
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = np.random.randint(0,255,(500,3))
            ## Parameters for lucas kanade optical flow
            lk_params = dict( winSize  = (15,15),
                              maxLevel = 3,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.03))



            ## Take first frame and find corners in it
            ret, old_frame = cap.read()

            old_frame, old_gray, p0 = self.IdentifyGoodFeatures(old_frame, blur, None)

            pi = p0
            lpi = len(pi)

            #print old_frame

            # Create a mask image for drawing purposes
            mask = np.zeros_like(old_frame)

            reevaluate = 0



            while(1):    
                ret,frame = cap.read()
                
                frame = cv2.GaussianBlur(frame, blur, 0)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
               
                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                if lpi > 1.3 * len (p1):
                    old_frame, old_gray, p0 = self.IdentifyGoodFeatures(frame, blur, None)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                    li = p1
                    lpi = len(li)
                    reevaluate+=1

                if p1 is not None:
                # Select good points
                    good_new = p1[st==1]
                    good_old = p0[st==1]

                    # draw the tracks
                    for i,(new,old) in enumerate(zip(good_new,good_old)):
                        a,b = new.ravel()
                        c,d = old.ravel()
                        cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                        cv2.circle(frame,(a,b),5,(200,200,250),-1)

                    cv2.putText(frame,'pi: ' + str(len(pi)),(5,25), font, 0.5,(255,255,25),2)
                    cv2.putText(frame,'p0: ' + str(len(p0)),(5,55), font, 0.5,(255,255,25),2)
                    cv2.putText(frame,'re: ' + str(reevaluate),(5,85), font, 0.5,(200,200,250),2)
                else:
                    cv2.putText(frame,'pi: 0 ',(5,55), font, 0.5,(255,255,25),2)

                cv2.imshow('frame',frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)

            cv2.destroyAllWindows()
            cap.release()
        



    def OpticalFlow2(self):

        cap = cv2.VideoCapture(0)
        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        while(1):
            ret, frame2 = cap.read()
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Change here
            horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
            vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
            Hhist = cv2.calcHist([horz],[0],None,[256],[0,256])

            horz = horz.astype('uint8')
            vert = vert.astype('uint8')

            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

            cv2.imshow('frame2',rgb)

            # Change here too
            cv2.imshow('Horizontal Component', horz)
            cv2.imshow('Vertical Component', vert)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png',frame2)
                cv2.imwrite('opticalhsv.png',rgb)
            prvs = next

        cap.release()
        cv2.destroyAllWindows()





#===================================================================================
    def runFaceIdentification(self,casfile,Radius, MaxInstances, AverageFactor):

        faceCascade = cv2.CascadeClassifier(casfile)

        video_capture = cv2.VideoCapture(0)

        estimator = clsMultipleObjectEstimation()

        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(gray,1.1,1,0|cv2.CASCADE_SCALE_IMAGE,(70,70))


            cv2.putText(frame,'objetos: ' + str(len(faces)) ,(5,25), font, 0.5,(255,255,25),2)
    
            #release current frame detection
            for est in estimator.List:   
                est.isInCurrentFrame = False


            
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                estimator.addNewObject(x+w/2,y+h/2, w, h, Radius,MaxInstances, AverageFactor)
                cv2.putText(frame, 'x=' + str(x) + ',y= ' + str(y) + ',w= ' + str(w) + ',h= ' + str(h) ,(x,y-10), font, 0.5,(255,255,255),2)
                #cv2.circle(frame,(x+w/2, y+h/2),50,(50,50,50),4,2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
     
   
            estimator.updateCurrentSceneObjects()
            

            i = 0
            v_faces = []
            if  estimator.List is not None:
                for est in estimator.List:     
                    if(est.Instances > 5):           
                        cv2.putText(frame,'objeto: ' + str(i) + "=" + str(est.Instances) + ", H= " + str(est.H),(5,50 + i*25), font, 0.5,(255,255,50),2)
                        #cv2.circle(frame,(est.Posx, est.Posy),Radius,(50,50,50),4,2)
                        cv2.circle(frame,(int(est.Posx), int(est.Posy)),int(est.radius),(250,250,50),4,2)
                        r = est.radius
                        v_faces.append(gray[est.Posy-r:est.Posy+r,est.Posx-r:est.Posx+r])
                        i += 1

            
            # Display the resulting frame
            cv2.imshow('Video', frame)

            i = 0
            for iface in v_faces:
                cv2.imshow('face ' + str(i),iface)
                i += 1



            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()