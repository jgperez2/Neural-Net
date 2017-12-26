# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 19:51:29 2017

@author: jgperez2
"""
import math
import skimage.io as io
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io

class inputNode(object):
    #Function V = F(u)
    v = 0.0
    #Error
    err = 0
    #Forward Nodes
    Next = []
    #Forward weights
    NextWt = []
    
    #get forward nodes and weights
    def backUp(self, node, wt):
        self.Next.append(node)
        self.NextWt.append(wt)
        
    def updateWeights(self):
        for i in range(0, len(self.Next)):
            self.Next[i].setU()
    
    def setError(self, err):
        for i in range(0, len(self.Next)):
                self.err += self.Next[i].err * self.NextWt[i]
        print self.err
        
    @staticmethod
    def getV(self):
        return self.v
        
    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except TypeError:
            return False
       
    #Is == next layer of nodes
    def __init__(self, V):
        self.v = V

class sigNode(object):
    #variable (u)
    u = 0.1
    #Function V = F(u)
    v = 0.1
    #Error
    err = 100
    #input v's, unweighted
    Is = []
    #weights for Is
    wts = []
    #Forward Nodes
    Next = []
    #Forward weights
    NextWt = []
    
    #get forward nodes and weights
    def backUp(self, node, wt):
        self.Next.append(node)
        self.NextWt.append(wt)
    
    def updateWeights(self):
        for i in range(0, len(self.Is)):
            self.wts[i + 1] = self.wts[i + 1] - (0.1 * (self.Is[i].err * self.Is[i].v))
    
    def setError(self, error):
        if (error == 0):
            for i in range(0, len(self.Next)):
                self.err += self.Next[i].err * self.NextWt[i]
            self.err = (self.v * (1 - self.v)) * self.err
        else:
            self.err = error
        print ("Error: %d" % self.err)
        self.giveSelf()
    
    def giveSelf(self):
        for i in range(0, len(self.Is)):
            self.Is[i].backUp(self, self.wts[i + 1])
    
    @staticmethod
    def getV(self):
        return self.v
    
    @staticmethod
    def getU(self):
        return self.u
    
    def setV(self):
        self.v = 1 / (1 - math.exp(-1 * self.u))
        
    def setU(self):
        self.u = 0
        for i in range(0, len(self.Is)):
            self.u += self.wts[0] + (self.wts[i+1] * self.Is[i].getV(self))
        
        self.setV()
    
    #initialize weights as 1 before training
    def setWeights(self, prevNodes):
        #initial weight for constant bias
        self.wts.append(0.1)
        #initial weights for inputs
        for x in range(0, len(prevNodes)):
            self.wts.append(0.1)
        self.setU()
       
    def __init__(self, Is):
        self.Is = Is
        self.setWeights(Is)
        
        
class CNN:
    
    def Error(nd, y):
        error = 0.5 * math.pow((nd.getV(nd) - y), 2)
        return error
    
    """
    Main
    """
    #input layer
    input = []
    
    img = io.imread('C:\\Python\ADF-10.tif')
    img_grayscale = rgb2gray(img)
    show_img = io.imshow(img)
    io.show()
    
    inputSQ = img
    
    for x in range(0, 2048):
        for o in range(0, 2048):
            input.append(inputSQ[x][o])
    inputnodes = []
    nodelvl1 = []
    nodelvl2 = []
    nodelvl3 = []
    
    #making input layer
    for x in range (0, len(input)):
        inputnodes.append(inputNode(input[x]))
    #making lvl 1 (fully connected)
    for x in range (0, 10):
        nodelvl1.append(sigNode(inputnodes))
    #making lvl 2 (fully connected)
    
    for x in range(0, 10):
        nodelvl2.append(sigNode(nodelvl1))
    #making lvl 3 (fully connected)
    for x in range(0, 10):
        nodelvl3.append(sigNode(nodelvl2))
        
    #making final nodes (fully connected)
    finalNode1 = sigNode(nodelvl3)
    finalNode2 = sigNode(nodelvl3)
    
    error1 = Error(finalNode1, 0)
    error2 = Error(finalNode2, 1)
    
    print ("Error1: %(1st)d, Error2: %(2nd)d" % {"1st": error1, "2nd": error2})
    
    finalNode1.setError(error1)
    finalNode2.setError(error2)
    
    #giving error to 3rd layer
    for x in range (0, len(nodelvl3)):
        nodelvl3[x].setError(0)
    #giving error to 2nd layer
    for x in range (0, len(nodelvl2)):
        nodelvl2[x].setError(0)
        #giving error to 1st layer
    for x in range (0, len(nodelvl1)):
        nodelvl1[x].setError(0)
        #giving error to input layer
    for x in range (0, len(inputnodes)):
        inputnodes[x].setError(0)
    
    #updating weights
    for x in range (0, len(nodelvl3)):
        nodelvl3[x].updateWeights()
    for x in range (0, len(nodelvl2)):
        nodelvl2[x].updateWeights()
    for x in range (0, len(nodelvl1)):
        nodelvl1[x].updateWeights()
    for x in range (0, len(inputnodes)):
        inputnodes[x].updateWeights()
    
    error1 = Error(finalNode1, 1)
    error2 = Error(finalNode2, 1)
    
    finalNode1.setError(error1)
    finalNode2.setError(error2)
    
    #giving error to 3rd layer
    for x in range (0, len(nodelvl3)):
        nodelvl3[x].setError(0)
    #giving error to 2nd layer
    for x in range (0, len(nodelvl2)):
        nodelvl2[x].setError(0)
        #giving error to 1st layer
    for x in range (0, len(nodelvl1)):
        nodelvl1[x].setError(0)
        #giving error to input layer
    for x in range (0, len(inputnodes)):
        inputnodes[x].setError(0)
    
    print ("Error1: %(1st)d, Error2: %(2nd)d" % {"1st": error1, "2nd": error2})