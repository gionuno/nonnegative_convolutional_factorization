#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:54:10 2017

@author: quien
"""
import numpy as np;
import numpy.random as rd;

import scipy.io as scio;
import matplotlib.pyplot as plt;
import matplotlib.image  as img;

from nncf import *;

L = 4;
K = 9;

a = scio.loadmat("frey_rawface.mat")['ff']/255.0;

X = np.zeros((200,28,20));
for i in range(X.shape[0]):
    X[i] = a[:,i].reshape((28,20));

test = learner(X,L*L,K);

it = 0;
while it < 100:
    it += 1;
    print "Iteration: "+str(it);
    test.step_W(10);
    
f,axarr = plt.subplots(L,L);
for i in range(L):
    for j in range(L):
        axarr[i,j].imshow(test.w[L*i+j],cmap='gray');
        axarr[i,j].set_xticklabels([]);
        axarr[i,j].set_yticklabels([]);
        axarr[i,j].grid(False)
plt.show()

H,f = test.step_H(X[1],100);
print f;
f,axarr = plt.subplots(L,L);
for i in range(L):
    for j in range(L):
        axarr[i,j].imshow(H[L*i+j],cmap='gray');
        axarr[i,j].set_xticklabels([]);
        axarr[i,j].set_yticklabels([]);
        axarr[i,j].grid(False)
plt.show()

s = np.zeros((X.shape[1],X.shape[2]));
f,axarr = plt.subplots(L,L);
for i in range(L):
    for j in range(L):
        s += conv2d(H[L*i+j],test.w[L*i+j],'full');
        axarr[i,j].imshow(s,cmap='gray');
        axarr[i,j].set_xticklabels([]);
        axarr[i,j].set_yticklabels([]);
        axarr[i,j].grid(False)
plt.show()

