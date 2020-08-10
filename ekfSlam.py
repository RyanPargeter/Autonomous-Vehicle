
import numpy as np
import argparse
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import time
import imutils
from scipy.spatial import distance
from typing import NamedTuple
import collections
from matplotlib import pyplot as plt
import os
from mpl_toolkits import mplot3d
import math
#%matplotlib inline
from IPython import display





class Slam:


    def __init__(self):
    
        self.q = np.array([0.0001, 0.0001])
        self.Q = np.diag(self.q**2)

        self.m = np.array([2*math.pi/180, .25])
        self.M = np.diag(self.m ** 2)

        self.R = np.array([[0],[0],[0]])
        self.u = np.array([[1], [0.0]])

        #initialize a landmark array

        self.y = np.zeros([2, 52])

        # estimator

        self.robotSize = 3
        self.x = np.zeros([self.robotSize + np.size(self.y), 1])
        self.P = np.zeros([np.size(self.x), np.size(self.x)])

        self.mapspace = np.arange(1,np.size(self.x))
        self.l = np.zeros([2, np.size(self.y)])

        self.r = np.nonzero(self.mapspace)[0][0:3]
        self.mapspace[self.r] = 0
        self.x[self.r] = self.R

        self.P[self.r[:, np.newaxis],self.r] = 0



    def backProject(self, r, y):
    
        p_r, PR_y = self.invScan(y)
        p, P_r, P_pr = self.fromFrame(r, p_r)
    
    
        P_y = P_pr @ PR_y 
    
        return p[0], p[1], P_r, P_y





    def invScan(self, y):
    
        d = y[0]
        a = y[1]
    
        px = d * math.cos(a)
        py = d* math.sin(a)
  
        p = np.vstack((px,py))
    
        P_y = np.vstack((np.hstack(([math.cos(a)], -d*math.sin(a))), np.hstack(([math.sin(a)], d*math.cos(a)))))
    
        return p, P_y




    def project(self, r, p):
    
        p_r, PR_r, PR_p = self.toFrame(r, p)
        y, Y_pr = self.scan(p_r)
    
        # chain rule
        Y_r = Y_pr @ PR_r
        Y_p = Y_pr @ PR_p
    
        return y, Y_r, Y_p






    def scan(self, x):
    
        px = x[0]
        py = x[1]
    
        d = math.sqrt(math.pow(px,2) + math.pow(py,2))
        a = math.atan2(py, px)
    
        y = np.array([[d], [a]])
    
        Y_x = np.vstack((np.hstack((px/math.pow((math.pow(px,2)+ math.pow(py,2)),(1/2)), py/math.pow((math.pow(px,2) + math.pow(py,2)),(1/2)))),
                       np.hstack((-py/(math.pow(px,2)*(math.pow(py,2)/math.pow(px,2) + 1)), 1/(px*(math.pow(py,2)/math.pow(px,2) + 1))))))
    
        return y, Y_x





    def toFrame(self,r, p):
    
        t = r[0:2]
        a = r[2]

        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

        p_r = R.conj().T @ (p - t)
    
    
        px = p[0]
        py = p[1]
        x = t[0]
        y = t[1]
    
        PR_r = np.vstack((np.hstack(([-math.cos(a), -math.sin(a)],   math.cos(a)*(py - y) - math.sin(a)*(px - x))), 
                          np.hstack(([math.sin(a), -math.cos(a)], - math.cos(a)*(px - x) - math.sin(a)*(py - y)))))
    
        PR_p = R.conj().T
    
    
    
        return p_r, PR_r, PR_p






    def robotMove(self, r, u):
    
        a = r[2]
        dx = u[0]
        da = u[1]
    
        ao = a + da
        dp = np.vstack((dx, [0])) # maybe transpose or make into rows?
    
        if ao > math.pi:
            ao = ao - 2*math.pi
        if ao < -math.pi:
            ao = ao + 2*math.pi
        
        to, TO_r, TO_dp = self.fromFrame(r, dp)
        AO_a = 1
        AO_da = 1
    
        RO_r = np.vstack((TO_r,[0, 0, AO_a]))
        RO_n = np.vstack((np.vstack((TO_dp[:,0], np.zeros([1,2]))) , [0, AO_da]))
        ro = np.vstack((to,ao)) 

        return ro, RO_r, RO_n





    def fromFrame(self, r, p_r):
        t = r[0:2]
        a = r[2]

    
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

        p = R@p_r + t

        px = p_r[0][0]
        py = p_r[1][0]

        P_r = np.vstack((np.hstack(([1, 0], -py*math.cos(a) -px*math.sin(a))), np.hstack((0, 1,   px*math.cos(a) -py*math.sin(a)))))
    
        P_pr = R
    
        return p, P_r, P_pr




