# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:01:23 2018

@author: Random
"""
import numpy as np

data = np.array([[0,0,0]])
for i in range(3):
    data[0][i] = (np.float32)(input("%d : " %i))
data.reshape([1,3])
print(data)
