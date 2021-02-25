import numpy
import math
import numpy as np
from skimage import util, exposure, data
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import pywt
import os

#Definir funciones de lectura de imágenes para mayor comodidad
def img_read(filename, mode = 'color'):
    if(mode == 'color'):   
        return cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2RGB)
    elif(mode=='grayscale' or mode=='greyscale' or mode == 'gray' or mode == 'grey'):
        return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        return None

def show_image(img):
    #img = cv2.imread(path)
    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

class features:
	def __init__(self,img1):
		self.contrast = []
		self.dissimilarity = []
		self.homogeneity = []
		self.energy = []
		self.correlation = []
		self.ASM = []
		#img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
		#img = img_as_ubyte(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
		glcm = greycomatrix(img1, [1], [np.pi/4, 0.75 * np.pi, 1.25*np.pi, 1.75*np.pi], 256, normed=True, symmetric=True)
		self.contrast.append(numpy.mean(greycoprops(glcm, 'contrast')[0]))
		self.dissimilarity.append(numpy.mean(greycoprops(glcm, 'dissimilarity')))
		self.homogeneity.append(numpy.mean(greycoprops(glcm, 'homogeneity')))
		self.energy.append(numpy.mean(greycoprops(glcm, 'energy')))
		self.correlation.append(numpy.mean(greycoprops(glcm, 'correlation')))
		self.ASM.append(numpy.mean(greycoprops(glcm, 'ASM')))

path = "Dataset/"

img_list = os.listdir(path)

Contraste = []
Disimilitud = []
Homogeneidad = []
Energia = []
Correlacion = []
ASM = []

for images in img_list:
	
	img = img_as_ubyte(img_read(path + images, 'gray'))
	img_features = features(img)
	img_features = features(img)
	Contraste = np.append(Contraste, img_features.contrast)
	Disimilitud = np.append(Disimilitud, img_features.dissimilarity)
	Homogeneidad = np.append(Homogeneidad, img_features.homogeneity)
	Energia = np.append(Energia, img_features.energy)
	Correlacion = np.append(Correlacion, img_features.correlation)
	ASM = np.append(ASM, img_features.ASM)

count, bins = np.histogram(Contraste, bins = 5)
print(count)
print(bins)




"""plt.figure(1)
plt.hist(Contraste, bins = 5)
plt.title('Contraste')
plt.show()

plt.figure(2)
plt.hist(Disimilitud, bins = 5)
plt.title('Disimilitud')
plt.show()

plt.figure(3)
plt.hist(Homogeneidad, bins = 5)
plt.title('Homogeneidad')
plt.show()

plt.figure(4)
plt.hist(Energia, bins = 5)
plt.title('Energia')
plt.show()

plt.figure(5)
plt.hist(Correlacion, bins = 5)
plt.title('Correlacion')
plt.show()

plt.figure(6)
plt.hist(ASM, bins = 5)
plt.title('ASM')
plt.show()"""