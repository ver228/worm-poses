#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:46:10 2019

@author: avelinojaver
"""

from smooth_manual_annotations import read_annotations_file 
import cv2
import matplotlib.pylab as plt


img_file = '/Users/avelinojaver/workspace/WormData/worm-poses/raw/rois/Phase5/mask/movie-941_worm-1_frame-11800.png'
annotations_file = '/Users/avelinojaver/workspace/WormData/worm-poses/raw/rois/Phase5/annotations/movie-941_worm-1_frame-11800.xml'


img = cv2.imread(img_file, -1)
pred = read_annotations_file(annotations_file)

plt.figure()
plt.imshow(img)

for pp in pred:
	for ss in pp:
		plt.plot(ss[:, 1], ss[:, 0], 'r.-')

plt.show()