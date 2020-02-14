#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:09:51 2020

@author: avelinojaver
"""

from link_skeletons import _process_file
import sys

if __name__ == '__main__':
    fname = sys.argv[1]
    
    src_postfix = '_unlinked-skels.hdf5'
    target_postfix = '_skeletonsNN.hdf5'
    
    
    save_file = fname[:-len(src_postfix)] + target_postfix
    
    _process_file(fname, save_file)