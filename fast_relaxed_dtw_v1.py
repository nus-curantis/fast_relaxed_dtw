#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division
import numbers
import numpy as np
from collections import defaultdict

try:
    range = xrange
except NameError:
    pass


def fast_relaxed_dtw(x, y, radius=1, dist=None, r=0):
    ''' return the approximate distance between 2 time series with O(N)
        time and memory complexity
        Parameters
        ----------
        x : array_like
        input array 1
        y : array_like
        input array 2
        radius : int
        size of neighborhood when expanding the path. A higher value will
        increase the accuracy of the calculation but also increase time
        and memory consumption. A radius equal to the size of x and y will
        yield an exact dynamic time warping calculation.
        dist : function or int
        The method for calculating the distance between x[i] and y[j]. If
        dist is an int of value p > 0, then the p-norm will be used. If
        dist is a function then dist(x[i], y[j]) will be used. If dist is
        None then abs(x[i] - y[j]) will be used.
        Returns
        -------
        distance : float
        the approximate distance between the 2 time series
        path : list
        list of indexes for the inputs x and y
        Examples
        --------
        >>> import numpy as np
        >>> import fastdtw
        >>> x = np.array([1, 2, 3, 4, 5], dtype='float')
        >>> y = np.array([2, 3, 4], dtype='float')
        >>> fastdtw.fastdtw(x, y)
        (2.0, [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)])
        '''
    x, y, dist = __prep_inputs(x, y, dist)
    return __fastdtw(x, y, radius, dist, r)


def __difference(a, b):
    return abs(a - b)


def __norm(p):
    return lambda a, b: np.linalg.norm(a - b, p)


def __fastdtw(x, y, radius, dist, r):
    min_time_size = radius + 2 + r
    
    if len(x) < min_time_size or len(y) < min_time_size:
        return relaxed_dtw(x, y, dist=dist, r=r)
    
    x_shrinked = __reduce_by_half(x)
    y_shrinked = __reduce_by_half(y)
    distance, path = \
        __fastdtw(x_shrinked, y_shrinked, radius=radius, dist=dist, r=r)
    window = __expand_window(path, len(x), len(y), radius)
    return __relaxed_dtw(x, y, window, dist=dist, r=r)


def __prep_inputs(x, y, dist):
    x = np.asanyarray(x, dtype='float')
    y = np.asanyarray(y, dtype='float')
    
    if x.ndim == y.ndim > 1 and x.shape[1] != y.shape[1]:
        raise ValueError('second dimension of x and y must be the same')
    if isinstance(dist, numbers.Number) and dist <= 0:
        raise ValueError('dist cannot be a negative integer')
    
    if dist is None:
        if x.ndim == 1:
            dist = __difference
        else:
            dist = __norm(p=1)
    elif isinstance(dist, numbers.Number):
        dist = __norm(p=dist)
    
    return x, y, dist


def relaxed_dtw(x, y, dist=lambda x,y : ((x-y)**2), r=0):
    ''' return the distance between 2 time series without approximation
        Parameters
        ----------
        x : array_like
        input array 1
        y : array_like
        input array 2
        dist : function or int
        The method for calculating the distance between x[i] and y[j]. If
        dist is an int of value p > 0, then the p-norm will be used. If
        dist is a function then dist(x[i], y[j]) will be used. If dist is
        None then abs(x[i] - y[j]) will be used.
        Returns
        -------
        distance : float
        the approximate distance between the 2 time series
        path : list
        list of indexes for the inputs x and y
        Examples
        --------
        >>> import numpy as np
        >>> import fastdtw
        >>> x = np.array([1, 2, 3, 4, 5], dtype='float')
        >>> y = np.array([2, 3, 4], dtype='float')
        >>> fastdtw.dtw(x, y)
        (2.0, [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)])
        '''
    x, y, dist = __prep_inputs(x, y, dist)
    return __relaxed_dtw(x, y, None, dist, r)


def __relaxed_dtw(x, y, window, dist, r):
    len_x, len_y = len(x), len(y)
    if window is None:
        window = [(i, j) for i in range(len_x) for j in range(len_y)]
    window = ((i + 1, j + 1) for i, j in window)
#    DTW = np.full((len(x), len(y)),np.inf)
    DTW = defaultdict(lambda: (float('inf'),))
#    DTW[0, 0] = (0, 0, 0)
    if (r==0):  ## classic DTW
        for m in range(len_x):
            DTW[m, 0] = (0, m, 0)
        for n in range(len_y):
            DTW[0, n] = (0, 0, n)
#        DTW[0:,0][0]= 0
#        DTW[0,0:][0] = 0
    else:
        for m in range(r+1):
            DTW[m, 0] = (0, m, 0)
            DTW[0, m] = (0, 0, m)
#        DTW[0:r,0][0] = 0
#        DTW[0,0:r][0] = 0
    for i, j in window:
        dt = dist(x[i-1], y[j-1])
        DTW[i, j] = min((DTW[i-1, j][0]+dt, i-1, j), (DTW[i, j-1][0]+dt, i, j-1),
                      (DTW[i-1, j-1][0]+dt, i-1, j-1), key=lambda a: a[0])
    i, j = len_x, len_y
    if (r!=0):
        min_x = DTW[len(x)-r,len(y)][0]
        min_y = DTW[len(x),len(y)-r][0]
        for m in range(r+1):
            min_x = min(DTW[len(x)-r+m,len(y)][0], min_x)
            min_y = min(DTW[len(x),len(y)-r+m][0], min_y)
        final_dist =min(min_x,min_y)
        if min_x < min_y:
            pre = DTW[len(x),len(y)-r]
            for m in range(r+1):
                if DTW[len(x),len(y)-r+m] < pre:
                    j = len(y)-r+m
        else:
            pre = DTW[len(x)-r,len(y)]
            for m in range(r+1):
                if DTW[len(x)-r+m,len(y)] < pre:
                    i = len(x)-r+m
    else:
        final_dist = (DTW[len(x),len(y)][0])
    path = []
    while not (i == 0 or j == 0):
#        print(DTW[i, j][0])
        path.append((i-1, j-1))
        i, j = DTW[i, j][1], DTW[i, j][2]
    path.reverse()
    return (final_dist, path)


def __reduce_by_half(x):
    return [(x[i] + x[1+i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]


def __expand_window(path, len_x, len_y, radius):
    path_ = set(path)
    for i, j in path:
        for a, b in ((i + a, j + b)
                     for a in range(-radius, radius+1)
                     for b in range(-radius, radius+1)):
            path_.add((a, b))

    window_ = set()
    for i, j in path_:
        for a, b in ((i * 2, j * 2), (i * 2, j * 2 + 1),
                     (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1)):
            window_.add((a, b))
                
            window = []
            start_j = 0
    for i in range(0, len_x):
        new_start_j = None
        for j in range(start_j, len_y):
            if (i, j) in window_:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j

    return window
