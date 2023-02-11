import cv2
import numpy as np


for k in range(2):
    picture = ('car.png','liberty.png')
    img = cv2.imread(picture[k])
    cv2.imshow("original",img)

    
# 灰階-------------------------------------
    size = img.shape

    height = size[0]

    width = size[1]

# RGB R=G=B = gray （R+G+B）/3

    d = np.zeros((height,width,1))

    for i in range(0,height):

        for j in range(0,width):

            (b,g,r) = img[i,j]

            b = int(b)

            g = int(g)

            r = int(r)

            gray = r*0.299+g*0.587+b*0.114

            d[i,j] = gray


    cv2.imshow('d',d.astype("uint8"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#銳化----------------------------------------------
    kernel = list([[0,-1,0],
               [-1,5,-1],
               [0,-1,0]])
    second = d.copy()
    convolution = np.pad(d,((1,1),(1,1),(0,0)),"constant",constant_values=0)
    for i in range(0,size[0]):
        for j in range(0,size[1]):
            x=(convolution[i,j]*kernel[0][0]
               +convolution[i,j+1]*kernel[0][1]
               +convolution[i,j+2]*kernel[0][2]
               +convolution[i+1,j]*kernel[1][0]
               +convolution[i+1,j+1]*kernel[1][1]
               +convolution[i+1,j+2]*kernel[1][2]
               +convolution[i+2,j]*kernel[2][0]
               +convolution[i+2,j+1]*kernel[2][1]
               +convolution[i+2,j+2]*kernel[2][2])
            if(x>255):
                x=255
            if(x<0):
                x=0
            second[i,j]=x
    
    cv2.imshow('Sharpended', second.astype("uint8"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#pooling--------------------------------------
    
    out = second.copy()
    G = 2
    new_height = int(img.shape[0] / G)
    new_weight = int(img.shape[1] / G)
    pooling = np.zeros((new_height,new_weight,1))
    
    
    for i in range(new_height):
        for j in range(new_weight):
            out[G*i:G*(i+1), G*j:G*(j+1)] = np.max(out[G*i:G*(i+1), G*j:G*(j+1)])
            pooling[i,j] = np.max(out[G*i:G*(i+1), G*j:G*(j+1)])
    cv2.imshow('pooling',pooling.astype("uint8"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

