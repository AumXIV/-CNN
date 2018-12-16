# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:23:59 2018

create_image.py
convert pdf file to image

@author: Admin
"""

import cv2
import glob
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg

from matplotlib import pyplot as plt
from pdf2image import convert_from_path, convert_from_bytes
from math import ceil, floor



def colect_image(pdf_dir, imageout_dir): #coonvert 1st page of PDF to image
    images = convert_from_path(pdf_dir, dpi=50)
    images[0].save(imageout_dir+'.png', 'png')
    
def colect_all_image(pdf_dir, imageout_dir): #coonvert all pages of PDF to images
    images = convert_from_path(pdf_dir, dpi=50)
    for i in range(len(images)):
            images[i].save(imageout_dir +'_'+str(i)+'.png', 'png')
            
def colect1image_folder(pdf_dir, imageout_dir): #coonvert 1st page of all PDF in folder to images
    for filename in glob.glob(pdf_dir + '*.pdf'):
        images = convert_from_path(pdf_dir + filename[len(pdf_dir):], dpi=50)
        images[0].save(imageout_dir + str(filename[len(pdf_dir):])+'.png', 'png')
        
def colect_all_image_folder(pdf_dir, imageout_dir): #coonvert all pages of all PDF in floder to images
    for filename in glob.glob(pdf_dir + '*.pdf'):
        images = convert_from_path(pdf_dir + filename[len(pdf_dir):], dpi=50)
        for i in range(len(images)):
            images[i].save(imageout_dir + str(filename[len(pdf_dir):])+'_'+str(i)+'.png', 'png')



         
