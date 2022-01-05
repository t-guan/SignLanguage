# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 23:36:00 2022

@author: Thomas Guan
"""
##Import Libraries
#We are going to preprocess all of the data, which are pixel-based data encoded to a letter, so need to perform the encoding here.
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import csv
