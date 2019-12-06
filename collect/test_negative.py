#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:48:17 2019

@author: avelinojaver
"""

from pathlib import Path
import gzip
import pickle

fname = Path.home() / 'workspace/WormData/worm-poses/labelled_rois/from_tierpsy_negative/negative_from_tierpsy.p.zip'
with gzip.GzipFile(fname, 'rb') as fid:
    negative_data = pickle.load(fid)