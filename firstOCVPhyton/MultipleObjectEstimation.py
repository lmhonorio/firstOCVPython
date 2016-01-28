import numpy as np
import math
import cv2 
from matplotlib import pyplot as plt


class clsObjectConfidence():


    def __init__(self,Posx = 0, Posy = 0, W = 0, H = 0, Instances = 0, Confidence = 0, isInCurrentFrame = False):
        self.Posx = Posx
        self.Posy = Posy
        self.W = W      
        self.H = H
        self.Instances = Instances
        self.Confidence = Confidence
        self.isInCurrentFrame = False        

    def distancefrom(self, Posx, Posy):
        return math.sqrt((self.Posx-Posx)**2 + (self.Posy-Posy)**2)

    @property
    def radius(self):
        return 0.6 * math.sqrt((self.H/2)**2 + (self.W/2)**2)



class clsMultipleObjectEstimation():

   
    def __init__(self):
        self.List = list()
        self.List.append(clsObjectConfidence())
    
    def addNewObject(self,Posx, Posy, W, H, distancethreshould, maxinstances, AverageFactor):

        isInlist = False;

        if self.List == None:
            self.List = list()
            self.List.append(clsObjectConfidence())

        for obj in self.List:
            obj.isInCurrentFrame = False;
            if obj.distancefrom(Posx,Posy) < distancethreshould:
                obj.isInCurrentFrame = True;
                isInlist = True;
                obj.Posx = (Posx + AverageFactor * obj.Posx)/(AverageFactor + 1)
                obj.Posy = (Posy + AverageFactor * obj.Posy)/(AverageFactor + 1)
                obj.W = (W + AverageFactor * obj.W)/(AverageFactor + 1)
                obj.H = (H + AverageFactor * obj.H)/(AverageFactor + 1)
                obj.Instances = min(maxinstances, obj.Instances + 2)

        if (not isInlist):
            nobj = clsObjectConfidence(Posx,Posy, W, H)
            nobj.Instances = 4;
            nobj.isInCurrentFrame = True;
            self.List.append(nobj)
            

    def updateCurrentSceneObjects(self):
           # notInFrame = [obj for obj in self.List if obj.isInCurrentFrame == False]

           for obj in self.List:
               if obj.Instances == 0:
                   self.List.remove(obj);
                   pass

           for obj in self.List:
               if obj.isInCurrentFrame == False:
                   obj.Instances = max(0, obj.Instances - 1)

                
            
