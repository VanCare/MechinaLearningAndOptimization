#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np

def my_func_5x(p):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    '''
    x1, x2,x3,x4,x5 = p
#     x = np.square(x1) + np.square(x2+1)
    x = np.square(x1) + np.square(x2+1) +np.square(x3+2) + np.square(x4+3)+np.square(x5+4)
    return x