#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:02:56 2017

@author: quien
"""

import numpy as np;
import numpy.random as rd;

import scipy.io as scio;
from scipy.signal import convolve2d as conv2d;
from scipy.signal import correlate2d as corr2d;

def get_conv(A,B):
    C = np.zeros((A.shape[1]+B.shape[1]-1,A.shape[2]+B.shape[2]-1));
    for l in range(A.shape[0]):
        C += conv2d(A[l],B[l],'full');
    return C;

def get_corr(A,B):
    C = np.zeros((B.shape[0],A.shape[0]-B.shape[1]+1,A.shape[1]-B.shape[2]+1));
    for l in range(C.shape[0]):
        C[l] = corr2d(A,B[l],'valid');
    return C;

def divergence(X,Y):
    return np.mean(X*np.log(X/Y)-X+Y);
def euclidean(X,Y):
    return np.mean((X-Y)**2);

#type_ = 'E' or 'D'
# E euclidean
# D divergence

class learner_euclidean:
    def __init__(self,X,L,K):
        self.X = X;
        self.K = K;
        self.L = L;
        self.w = rd.rand(L,K,K);
    
    def step_H(self,x,T):
        H = rd.rand(self.L,x.shape[0]-self.K+1,x.shape[1]-self.K+1);
        for t in range(T):
            y = get_conv(H,self.w);
            numH = get_corr(x,self.w);
            denH = get_corr(y,self.w);
            H = H * numH / denH;
        return H, euclidean(x,get_conv(H,self.w));
    
    def step_W(self,T):
        numw = np.zeros(self.w.shape);
        denw = np.zeros(self.w.shape);
        meanf = 0.0;
        n = 0;
        for x in self.X:
            H,f = self.step_H(x,T);
            numw = (n*numw+get_corr(x,H))/(n+1.0);
            denw = (n*denw+get_corr(get_conv(H,self.w),H))/(n+1.0);
            meanf = (n*meanf+f)/(n+1.0);
            n += 1;
            print n;
        print "Mean Error: ",meanf;
        self.w = self.w * numw / denw;


class learner_divergence:
    def __init__(self,X,L,K):
        self.X = X;
        self.K = K;
        self.L = L;
        self.w = rd.rand(L,K,K);
        self.o = np.ones(X[0].shape);
    
    def step_H(self,x,T):
        H = rd.rand(self.L,x.shape[0]-self.K+1,x.shape[1]-self.K+1);
        denH = get_corr(self.o,self.w);
        for t in range(T):
            y = get_conv(H,self.w);
            numH = get_corr(x / y, self.w);
            H = H * numH / denH;
        return H, divergence(x,get_conv(H,self.w));
    
    def step_W(self,T):
        numw = np.zeros(self.w.shape);
        denw = np.zeros(self.w.shape);
        meanf = 0.0;
        n = 0;
        for x in self.X:
            H,f = self.step_H(x,T);
            y = get_conv(H,self.w);
            numw = (n*numw+get_corr(x / y, H))/(n+1.0);
            denw = (n*denw+get_corr(self.o,H))/(n+1.0);
            meanf = (n*meanf+f)/(n+1.0);
            n += 1;
            print n;
        print "Mean Error: ",meanf;
        self.w = self.w * numw / denw;
