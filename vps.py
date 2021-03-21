# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:43:32 2021

@author: jesse
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

listFileNames = []
numOfImages = 30
fileName = 'C:\\Users\\jesse\\Google Drive\\School\\usf\\ComputerVision\\Group\\archive\\images\\images\\train\\clean\\'
listImages = []
listGrayImages = []
listEdges = []
minLineLength = 20
maxLineGap = 5
listLines = []
listEuclidPoints = []
listEuclidPointsAllImgs = []

# Generate Perspective Lines
for i in range(numOfImages):
    listFileNames.append(fileName + str(i) + '.png')
    listImages.append(cv.imread(listFileNames[i]))
    listGrayImages.append(cv.cvtColor(listImages[i], cv.COLOR_BGR2GRAY))
    listEdges.append(cv.Canny(listGrayImages[i],50,150,apertureSize = 3))
    listLines.append(cv.HoughLines(listEdges[i], 1, np.pi / 180, 100, minLineLength, maxLineGap))
    #print('222    listFileNames[',i,'] ', listFileNames[i])
    #print('222    listLines[i].shape ', listLines[i].shape)

    listPolarPoints = [] # (rho, theta)   for HoughLines
    for j in range(len(listLines[i])):
        for k in range(len(listLines[i][j])):
          #  print('222          listLines[',i,'][',j,'] ', listLines[i][j])
          #  print('222          listLines[',i,'][',j,'].shape ', listLines[i][j].shape)
          #  print('222          listLines[',i,'][',j,'][k] ', listLines[i][j][k])


           # cv.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
            listPolarPoints.append((listLines[i][j][k][0], listLines[i][j][k][1]))
    #print('333    len(listPolarPoints) ', len(listPolarPoints))
    #print('333    listPolarPoints ', listPolarPoints)

    listEuclidPoints = []
    for j in range(len(listPolarPoints)):
        rho = listPolarPoints[j][0]
        theta = listPolarPoints[j][1]
        #print('444      listLines[',i,'] ', listLines[i])
     #   print('444          rho ', rho)
     #   print('444          theta ', theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))
        listEuclidPoints.append((x1, y1, x2, y2))
        #cv.line(listImages[i],(x1,y1),(x2,y2),(0,255,0),1)
    
    listEuclidPointsAllImgs.append(listEuclidPoints)

    #cv.imwrite('hougher' + str(i) + '.png', listImages[i]) 
    
print('000      len(listFileNames) ', len(listFileNames))
print('000      len(listImages) ', len(listImages))
print('000      len(listEuclidPoints) ', len(listEuclidPoints))
print('000      len(listEuclidPointsAllImgs) ', len(listEuclidPointsAllImgs))

#print('000      listEuclidPoints ', listEuclidPoints)

listGroupLines = []
listGroupLinesAllImgs = []
groupThreshInPixelsX = 40
groupThreshInPixelsY = 20

avgX = -1000000
avgY = -1000000
listAvgX = []
listAvgY = []
# Group similar lines together
# Could probably use K Clustering here
for i in range(len(listEuclidPointsAllImgs)):   # len(listEuclidPointsAllImgs)
    print('777    len(listEuclidPointsAllImgs[', i, ']  ', len(listEuclidPointsAllImgs[i]))
    
    listGroupLines = []
    for j in range(len(listEuclidPointsAllImgs[i])):
        print('777    len(listEuclidPointsAllImgs[', i, '][',j,'] ', len(listEuclidPointsAllImgs[i][j]))

        x1 = listEuclidPointsAllImgs[i][j][0]
        y1 = listEuclidPointsAllImgs[i][j][1]
        x2 = listEuclidPointsAllImgs[i][j][2]
        y2 = listEuclidPointsAllImgs[i][j][3]
        
        if j == 0:
            avgX = x1
            avgY = y1
            continue
        # Using abs val may be an issue here
        diffX = abs(x1 - avgX)
        diffY = abs(y1 - avgY)
        
       # print('777           i ', i)
       # print('777           x1 ', x1, ' y1 ', y1, ' x2 ', x2, ' y2 ', y2)
       # print('777          diffX ', diffX)
       # print('777          diffY ', diffY)
        if diffX < groupThreshInPixelsX:
            print('777    diffX < groupThreshInPixelsX      diffX ', diffX)
            if diffY < groupThreshInPixelsY:
                print('777    diffY < groupThreshInPixelsX      diffY ', diffY)
                listGroupLines.append(listEuclidPointsAllImgs[i][j])
                
    listGroupLinesAllImgs.append(listGroupLines)
            
            
print('888      len(listGroupLines) ', len(listGroupLines))
print('888      len(listGroupLinesAllImgs) ', len(listGroupLinesAllImgs))
print('888      listGroupLinesAllImgs ', listGroupLinesAllImgs)
# Each i here corresponds to an image
for i in range(len(listGroupLinesAllImgs)):
    if len(listGroupLinesAllImgs[i]) == 0:
        continue
   
    print('888      listGroupLinesAllImgs[', i, '] ', listGroupLinesAllImgs[i])

    for j in range(len(listGroupLinesAllImgs[i])):
       # print('999  groupys    listGroupLinesAllImgs[', i, '][', j, '] ', listGroupLinesAllImgs[i][j])

        x1 = listGroupLinesAllImgs[i][j][0]
        y1 = listGroupLinesAllImgs[i][j][1]
        x2 = listGroupLinesAllImgs[i][j][2]
        y2 = listGroupLinesAllImgs[i][j][3]
        plt.scatter(x1, y1) 
        #cv.line(listImages[i],(x1,y1),(x2,y2),(0,0,255),1)
    #cv.imwrite('hougherFewer' + str(i) + '.png', listImages[i]) 
       # plt.plot(x1, y1)
        #plt.plot(x2, y2, 'ro')
    #plt.show()
    
#plt.scatter(2, 3)    
plt.show()
        




    
    
    
    
    
    

#img = cv.imread('C:\\Users\\jesse\\Google Drive\\School\\usf\\ComputerVision\\Group\\archive\\images\\images\\train\\clean\\31.png')


#gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#edges = cv.Canny(gray,50,150,apertureSize = 3)

#lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#lines = cv.HoughLines(edges,1,np.pi/180,100,minLineLength,maxLineGap)

#print('111       type(lines) ', type(lines))
#print('111       lines.shape ', lines.shape)
#print('111       lines[0] ', lines[0])
#print('111       lines[0] shape ', lines[0].shape)


#listLines = [] # (rho, theta)   for HoughLines
#for i in range(len(lines)):
#    for j in range(len(lines[i])):
        #print('222          lines[',i,'][',j,'] ', lines[i][j])
        
       # cv.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
#        listLines.append((lines[i][j][0], lines[i][j][1]))
'''
print('333       len(listLines) ', len(listLines))

for i in range(len(listLines)):
    rho,theta = listLines[i]
    #print('444      listLines[',i,'] ', listLines[i])
    #print('444          rho ', rho)
    #print('444          theta ', theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 10000*(-b))
    y1 = int(y0 + 10000*(a))
    x2 = int(x0 - 10000*(-b))
    y2 = int(y0 - 10000*(a))
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),1)

cv.imshow('hougher', img)   
    

# fine tune parameters
listLinesFine = [] # (rho, theta)   for HoughLines
linesFine = cv.HoughLines(edges, 0.7, np.pi/120, 120, min_theta=np.pi/36, max_theta=np.pi-np.pi/36)


for i in range(len(linesFine)):
    for j in range(len(linesFine[i])):
        #print('222          lines[',i,'][',j,'] ', lines[i][j])
        
       # cv.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        listLinesFine.append((linesFine[i][j][0], linesFine[i][j][1]))

for i in range(len(listLinesFine)):
    rho,theta = listLinesFine[i]
    # skip near-vertical lines
    if abs(theta-np.pi/90) < np.pi/9:
        continue
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 10000*(-b))
    y1 = int(y0 + 10000*(a))
    x2 = int(x0 - 10000*(-b))
    y2 = int(y0 - 10000*(a))
    #cv.line(img,(x1,y1),(x2,y2),(0,255,0),1)
    
#cv.imshow('finer', img)

print('555       len(listLinesFine) ', len(listLinesFine))

# Maybe proceed with regression here if needed to find correct lines

'''



#print('111       lines ', lines)
'''
listLines = []     # (x0, y0, x1, y1)    line endpoints for HoughLinesP
for i in range(len(lines)):
    for j in range(len(lines[i])):
        #print('222          lines[',i,'][',j,'] ', lines[i][j])
        x0 = lines[i][j][0]
        y0 = lines[i][j][1]
        x1 = lines[i][j][2]
        y1 = lines[i][j][3]
        cv.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        listLines.append((x0, y0, x1, y1))


print('333       len(listLines) ', len(listLines))
'''
#cv.imshow('hougher', img)