from balu.ImagesAndData import balu_imageload
from balu.ImageProcessing import Bim_segbalu
from balu.FeatureExtraction import Bfx_haralick
from balu.InputOutput import Bio_printfeatures
from balu.InputOutput import Bio_plotfeatures
from balu.ImagesAndData import balu_load
import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
from PIL import Image
from skimage import util, exposure, data
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from mahotas.colors import rgb2gray
from balu.ImageProcessing import Bim_segbalu
from balu.ImagesAndData import balu_imageload
from balu.FeatureExtraction import Bfx_basicint
from balu.ImagesAndData import balu_imageload
from balu.FeatureExtraction import Bfx_lbp
from balu.InputOutput import Bio_printfeatures
from matplotlib.pyplot import bar, figure, show
from balu.FeatureExtraction import Bfx_fitellipse
from balu.ImagesAndData import balu_imageload
from balu.FeatureExtraction import Bfx_basicgeo
from balu.FeatureExtraction import Bfx_fourierdes
from skimage.morphology import label
from matplotlib.pyplot import imshow, show
from balu.FeatureExtraction import Bfx_hugeo
from balu.FeatureExtraction import Bfx_all


path = r"G:\Unidades compartidas\SAIC Ingeniería\Proyecto Aserrio\Dataset_texturas\Dataset"
img_list = os.listdir(path)


F1=[]
F2=[]
F3=[]
F4=[]
F5=[]
F6=[]
F7=[]
F8=[]
F9=[]
F10=[]
F11=[]
F12=[]
F13=[]
F14=[]
F15=[]
F16=[]
F17=[]
F18=[]

# Features Basic intensity features
for images in img_list:
    options = {'show': True, 'mask': 5}   # Gauss mask for gradient computation and display results
    I = io.imread(path + chr(92) + images)      #input image                        #green channel
    dim=(75,75)
    cv2.resize(I, dim, interpolation = cv2.INTER_LANCZOS4)
    I=cv2.cvtColor(I,cv2.COLOR_BGR2YUV)
    I = I[:,:,2]
    I= cv2.convertScaleAbs(I, alpha = -1, beta =105)
         # region of interest (green)
    R,_,_ = Bim_segbalu(I);                     # segmentation
    X, Xn = Bfx_basicint(I,R,options)
    Bio_printfeatures(X, Xn)
    
    Feature1=np.append(F3,X[0][2])
    Feature2=np.append(F5,X[0][4])




### Features HARALICK ####


for images in img_list:
    options = {'dharalick': [1, 2, 3, 4, 5]}   # Gauss mask for gradient computation and display results
    I = io.imread(path + chr(92) + images)      #input image                        #green channel
    dim=(75,75)
    cv2.resize(I, dim, interpolation = cv2.INTER_LANCZOS4)
    I=cv2.cvtColor(I,cv2.COLOR_BGR2YUV)
    I = I[:,:,2]
    I= cv2.convertScaleAbs(I, alpha = -1, beta =105)  
    
    #segmentation                         #green channel
    X, Xn = Bfx_haralick(I, None, options)     #Haralick features
    Bio_printfeatures(X, Xn)    


    Feature3=np.append(F1,X[0][0]) 
    Feature4=np.append(F2,X[0][1])  
    Feature5=np.append(F3,X[0][2])   
    Feature6=np.append(F4,X[0][3])  
    Feature7=np.append(F5,X[0][4])  
    Feature8=np.append(F6,X[0][5])  
    Feature9=np.append(F7,X[0][6])  
    Feature10=np.append(F8,X[0][7])  
    Feature11=np.append(F9,X[0][8])  
    Feature12=np.append(F10,X[0][9])   
    Feature13=np.append(F11,X[0][10])  
    Feature14=np.append(F12,X[0][11])  
    Feature15=np.append(F13,X[0][12])  
    Feature16=np.append(F14,X[0][13])  


### Caracteristicas  Standard geometric features of a binary image R ####

for images in img_list:
    I=io.imread(path + chr(92) + images)# input image
    dim=(75,75)
    cv2.resize(I, dim, interpolation = cv2.INTER_LANCZOS4)
    I=cv2.cvtColor(I,cv2.COLOR_BGR2YUV)
    I = I[:,:,2]
    I= cv2.convertScaleAbs(I, alpha = -1, beta =105) 
                    # segmentation
    X, Xn = Bfx_basicgeo(I)               # Fourier descriptors
    Bio_printfeatures(X, Xn)

    Feature17=np.append(F5,X[0][4])  
    Feature18=np.append(F6,X[0][5])    
    Feature19=np.append(F8,X[0][7])  
    Feature20=np.append(F9,X[0][8])    
    Feature21=np.append(F14,X[0][13])  
    Feature22=np.append(F15,X[0][14]) 
    Feature23=np.append(F16,X[0][15]) 
    Feature24=np.append(F17,X[0][16]) 


###### FEATURES  ####
Feature1=Feature1
Feature2=Feature2
Feature3=Feature3
Feature4=Feature4
Feature5=Feature5
Feature6=Feature6
Feature7=Feature7
Feature8=Feature8
Feature9=Feature9
Feature10=Feature10
Feature11=Feature11
Feature12=Feature12
Feature13=Feature13
Feature14=Feature14
Feature15=Feature15
Feature16=Feature16
Feature17=Feature17
Feature18=Feature18
Feature19=Feature19
Feature20=Feature20
Feature21=Feature21
Feature22=Feature22
Feature23=Feature23
Feature24=Feature24





