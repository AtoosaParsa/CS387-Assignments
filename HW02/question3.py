#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:20:49 2021

@author: atoosa
"""
import sys
import numpy as np
import pandas as pd

# uniform distribution
def uniformDistr(a, b, size):
    data = np.random.uniform(low=a, high=b, size=size)
    return data

def uniformMoments(a, b):
    m1 = (a+b)/2
    m2 = ((b-a)**2)/12
    return m1, m2

# exponential distribution
def expDistr(beta, size):
    data = np.random.exponential(scale=beta, size=size)
    return data

def expMoments(beta):
    m1 = beta
    m2 = beta ** 2
    return m1, m2

# normal distribution
def normalDistr(mue, sigma, size):
    data = np.random.normal(loc=mue, scale=sigma, size=size)
    return data

def normalMoments(mue, sigma):
    m1 = mue
    m2 = sigma ** 2
    return m1, m2

# power-law distribution
# https://stats.stackexchange.com/questions/173242/random-sample-from-power-law-distribution
# https://arxiv.org/pdf/0706.1062.pdf : equation D4
# using inverse of CDF
def powerlawDistr(gamma, a, size):
    r = np.random.uniform(low=0, high=1, size=size)
    data = a * (1-r)**(-1/(gamma-1))
    return  data

def powerlawMoments(gamma, a):
    if gamma>2:
        m1 = a * ((1-gamma)/(2-gamma))
    else:
        m1 = np.inf
    if gamma>3:
        m2 = a**2 * (gamma-1) * ((1/(gamma-3))-((gamma-1)/((gamma-2)**2)))
    else:
        m2 = np.inf
    
    return m1, m2

np.random.seed(10)

# uniform
# calculate sample moments
print("uniform:")
a = 1
b = 10
for N in [1, 10, 100, 1000, 10000]:
    print("N: " + str(N))
    sample_m1 = []
    sample_m2 = []
    for K in range(0, 30):
        data = uniformDistr(a, b, N)
        sample_m1.append(np.mean(data))
        sample_m2.append(np.var(data))
    m1 = np.mean(sample_m1)
    m2 = np.mean(sample_m2)
    print("sample moments: "+ str(m1) + " ,"+ str(m2))
    
# calculate moments from formula    
true_m1, true_m2  = uniformMoments(a, b)
print("calculated moments: "+ str(true_m1) + " ,"+ str(true_m2))

# expoonential
# calculate sample moments
print("exponential:")
beta = 1
for N in [1, 10, 100, 1000, 10000]:
    print("N: " + str(N))
    sample_m1 = []
    sample_m2 = []
    for K in range(0, 30):
        data = expDistr(beta, N)
        sample_m1.append(np.mean(data))
        sample_m2.append(np.var(data))
    m1 = np.mean(sample_m1)
    m2 = np.mean(sample_m2)
    print("sample moments: "+ str(m1) + " ,"+ str(m2))
    
# calculate moments from formula    
true_m1, true_m2  = expMoments(beta)
print("calculated moments: "+ str(true_m1) + " ,"+ str(true_m2))

# normal
# calculate sample moments
print("normal:")
mue = 1
sigma = 10
for N in [1, 10, 100, 1000, 10000]:
    print("N: " + str(N))
    sample_m1 = []
    sample_m2 = []
    for K in range(0, 30):
        data = normalDistr(mue, sigma, N)
        sample_m1.append(np.mean(data))
        sample_m2.append(np.var(data))
    m1 = np.mean(sample_m1)
    m2 = np.mean(sample_m2)
    print("sample moments: "+ str(m1) + " ,"+ str(m2))
    
# calculate moments from formula    
true_m1, true_m2  = normalMoments(mue, sigma)
print("calculated moments: "+ str(true_m1) + " ,"+ str(true_m2))

# powerlaw
# calculate sample moments
print("power-law:")
a = 1
gamma = 4
for N in [1, 10, 100, 1000, 10000]:
    print("N: " + str(N))
    sample_m1 = []
    sample_m2 = []
    for K in range(0, 30):
        data = powerlawDistr(gamma, a, N)
        sample_m1.append(np.mean(data))
        sample_m2.append(np.var(data))
    m1 = np.mean(sample_m1)
    m2 = np.mean(sample_m2)
    print("sample moments: "+ str(m1) + " ,"+ str(m2))
    
# calculate moments from formula    
true_m1, true_m2  = powerlawMoments(gamma, a)
print("calculated moments: "+ str(true_m1) + " ,"+ str(true_m2))