# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:43:32 2021

@author: jesse
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.axisartist.axislines import Subplot



listFileNames = []
fileName = 'C:\\Users\\jesse\\Google Drive\\School\\usf\\ComputerVision\\Group\\images\\original\\'
listImages = []
listGrayImages = []
listEdges = []
minLineLength = 80
maxLineGap = 2
listLines = [] # A vector that will store the parameters (r,θ) of the detected lines
listEuclidPoints = [] # (x1, y1, x2, y2)   endpoints of lines
listEuclidPointsAllImgs = [] # (x1, y1, x2, y2)   endpoints of lines for all images
listImgHeight = []
listImgWidth = []

HOUGH_THRESHOLD = 100 # The minimum number of intersections to "*detect*" a line
NUM_OF_CLUSTERS = 8


NUM_OF_IMAGES = 5
'''
 #  Standard Hough Line Transform
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
with the following arguments:
dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
lines: A vector that will store the parameters (r,θ) of the detected lines
rho : The resolution of the parameter r in pixels. We use 1 pixel.
theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
threshold: The minimum number of intersections to "*detect*" a line
srn and stn: Default parameters to zero. Check OpenCV reference for more info.
'''

fileNameStrBeforeNum = 'Copy of '
fileExtJpg = '.jpg'
fileExtPng = '.png'
# Generate Perspective Lines
for i in range(NUM_OF_IMAGES):
    fillName = fileName + fileNameStrBeforeNum + str(i) + fileExtPng
    print('0000000000       fillName   ', fillName)
    listFileNames.append(fillName)
    listImages.append(cv.imread(listFileNames[i]))
    print('0000000000       listImages[i].shape   ', listImages[i].shape)
    listImgHeight.append(listImages[i].shape[0])
    listImgWidth.append(listImages[i].shape[1])
    print('0000000000       listImgWidth[', i, '] ', listImgWidth[i])
    print('0000000000       fillName   ', fillName)

for i in range(NUM_OF_IMAGES):
    listGrayImages.append(cv.cvtColor(listImages[i], cv.COLOR_BGR2GRAY))
    
for i in range(NUM_OF_IMAGES):    
    listEdges.append(cv.Canny(listGrayImages[i],50,150,apertureSize = 3))
    
for i in range(NUM_OF_IMAGES):
    listLines.append(cv.HoughLines(listEdges[i], 1, np.pi / 180, HOUGH_THRESHOLD))
    print('0000000000       listLines[i]   ', listLines[i])

    #print('222    listFileNames[',i,'] ', listFileNames[i])
    #print('222    listLines[i].shape ', listLines[i].shape)
for i in range(NUM_OF_IMAGES):
    listPolarPoints = [] # (rho, theta)   for HoughLines
    if listLines[i] is not None:
        for j in range(len(listLines[i])):
            for k in range(len(listLines[i][j])): # k is set of two points
                #print('222          listLines[',i,'][',j,'] ', listLines[i][j])
                #print('222          listLines[',i,'][',j,'].shape ', listLines[i][j].shape)
                #print('222          listLines[',i,'][',j,'][k] ', listLines[i][j][k])
    
    
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
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            listEuclidPoints.append((x1, y1, x2, y2))
            #cv.line(listImages[i],(x1,y1),(x2,y2),(0,255,0),1)
        
        listEuclidPointsAllImgs.append(listEuclidPoints)

    #cv.imwrite('hougher' + str(i) + '.png', listImages[i]) 
    
print('000      len(listFileNames) ', len(listFileNames))
print('000      len(listImages) ', len(listImages))
print('000      len(listEuclidPoints) ', len(listEuclidPoints))
print('000      len(listEuclidPointsAllImgs) ', len(listEuclidPointsAllImgs))
print('000      len(listEuclidPointsAllImgs[0]) ', len(listEuclidPointsAllImgs[0]))




#kInputEndPointsPreAsNp = np.asarray(listEuclidPointsAllImgs)
kInputEndPointsPre = []
kInputSlopesPre = []
listKInputEndPoints = []
listKInputSlopes = []
for h in range(len(listEuclidPointsAllImgs)):
    for i in range(len(listEuclidPointsAllImgs[h])):
        print('000 000   listEuclidPointsAllImgs[i] ', listEuclidPointsAllImgs[h][i])
        x1 = listEuclidPointsAllImgs[h][i][0]
        y1 = listEuclidPointsAllImgs[h][i][1]
        x2 = listEuclidPointsAllImgs[h][i][2]
        y2 = listEuclidPointsAllImgs[h][i][3]
        diffY = y2 - y1
        diffX = x2 - x1
        if diffX != 0:
            slopePre = int(float(diffY) / float(diffX))
        else:
            slopePre = 100
        kInputEndPointsPre.append((x1, y1))
        kInputEndPointsPre.append((x2, y2))
        kInputSlopesPre.append(slopePre)
        
        
    kInput = np.asarray(kInputEndPointsPre)
    kInputSlope = np.asarray(kInputSlopesPre)
    #print('111   before  kInput shape ', kInput.shape)
    #print('111   before  kInput type ', type(kInput))
    #print('111   before  kInput ', kInput)
    #print('111   before  kInputSlope.shape ', kInputSlope.shape)
    listKInputEndPoints.append(kInput)
    listKInputSlopes.append(kInputSlope)


listKMeansEndPoint = []
listClusterEndPoint = []
listLabelsEndPoint = []

listKMeansSlope = []
listClusterSlope = []
listLabelsSlope = []
for i in range(NUM_OF_IMAGES):
    # produce k means info for endpoints
    kmeansEndPoint = KMeans(n_clusters = NUM_OF_CLUSTERS, random_state = 0).fit(listKInputEndPoints[i]) # end points of lines
    clusterEndPoint = kmeansEndPoint.cluster_centers_
    labelsEndPoint = kmeansEndPoint.labels_
    
    listKMeansEndPoint.append(kmeansEndPoint)
    listClusterEndPoint.append(clusterEndPoint)
    listLabelsEndPoint.append(labelsEndPoint)
    
    # produce k means info for slopes
    kmeansSlope = KMeans(n_clusters = NUM_OF_CLUSTERS, random_state = 0).fit(listKInputSlopes[i].reshape(-1, 1)) # slopes of lines, reshape is for one feature
    clusterSlope = kmeansSlope.cluster_centers_
    labelsSlope = kmeansSlope.labels_
    
    listKMeansSlope.append(kmeansSlope)
    listClusterSlope.append(clusterSlope)
    listLabelsSlope.append(labelsSlope)
    

# generate color list
listColor = [];
inc = 255.0 / NUM_OF_CLUSTERS
rgbDivide = int(255.0 / (NUM_OF_CLUSTERS * 3.0))
col = 0
isTopColor = True
for i in range(NUM_OF_CLUSTERS):
    if (isTopColor):
        if i < rgbDivide:
            col = (int(i * inc), 0, 255 - int(i * inc))
        elif i >= rgbDivide and i < rgbDivide * 2:
            col = (0, int(i * inc), 0)
        elif i >= rgbDivide * 2 and i < rgbDivide * 3:
            col = (int(i * inc), 0, 0)
        
        isTopColor = False
    else:
        if i < rgbDivide:
            col = (0, int(i * inc), 0)
        elif i >= rgbDivide and i < rgbDivide * 2:
            col = (int(i * inc),  int(i * inc), 255 - int(i * inc))
        elif i >= rgbDivide * 2 and i < rgbDivide * 3:
            col = (255 - int(i * inc), 0, 0)
        isTopColor = True
    
    
    listColor.append(col)
    
# generate hex color list
listColorHex = []
listColorHex.append('#fa0f00')
listColorHex.append('#FADC00')
listColorHex.append('#A3FA00')
listColorHex.append('#0FFA00')
listColorHex.append('#00FACD')
listColorHex.append('#00AAFA')
listColorHex.append('#0026FA')
listColorHex.append('#8500FA')
    
print('333     9999         len  listColor ', len(listColor))
print('333     9999           listColor ', listColor)
    
# print display 
print('Display    listClusterSlope ', listClusterSlope)

# histograms of clusters
clusterHistHeight = 3
clusterHistWidth = 3
indexListClusterSlope = -1
fig, ax = plt.subplots(3, 3, tight_layout=True)
for i in range(clusterHistWidth):
    for j in range(clusterHistHeight):
        indexListClusterSlope += 1
        if indexListClusterSlope < NUM_OF_IMAGES:
            ax[i][j].hist(listClusterSlope[indexListClusterSlope], bins = NUM_OF_CLUSTERS)

# draw clustered slopes on images
for i in range(NUM_OF_IMAGES):
    for j in range(len(listClusterSlope[i])):
        
        sloper = listClusterSlope[i][j]
        x1 = 0.0
        y1 = 0.0
        x2 = 0.0
        y2 = 0.0
        #print('222   sloper ',sloper)
        if sloper > 0:
            x1 = 0.0
            y1 = 0.0
            yInter = float(listImgHeight[i])
            x2 = float(listImgWidth[i])
            y2 = float(sloper * x2) + yInter
        elif sloper < 0:
            x1 = 0.0
            y1 = 0.0 
            yInter = 0.0
            x2 = float(listImgWidth[i])
        elif sloper == 100:
            x1 = float(listImgWidth[i]) / 2.0
            y1 = 0.0
            x2 = x1
            y2 = float(sloper * x2)
        else:
            x1 = 0.0
            y1 = 0.0
            x2 = float(listImgWidth[i]) / 2.0
            y2 = float(sloper * x2) + yInter
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        
        cv.line(listImages[i], (x1, y1), (x2, y2), listColor[i], 2)
        
    cv.imwrite('C:\\Users\\jesse\\Google Drive\\School\\usf\\ComputerVision\\Group\\images\\Display\\slopeclusters\\slopes' + str(i) + '.png', listImages[i]) 




# print display endpoints
print('Display    listClusterEndPoint ', listClusterEndPoint)

# histograms of clusters endpoints
clusterHistHeight = 3
clusterHistWidth = 3
indexListClusterEnd = -1
fig, ax = plt.subplots(3, 3, tight_layout=True)
for i in range(clusterHistWidth):
    for j in range(clusterHistHeight):
        indexListClusterEnd += 1
        if indexListClusterEnd < NUM_OF_IMAGES:
            for k in range(0, len(listClusterEndPoint[indexListClusterEnd]), 2): 
                ax[i][j].scatter(listClusterEndPoint[indexListClusterEnd][k][0], listClusterEndPoint[indexListClusterEnd][k][1], c = listColorHex[k])

# draw clustered endpoints on images
for i in range(NUM_OF_IMAGES):
    for j in range(len(listClusterEndPoint[i])):
        
        sloper = listClusterSlope[i][j]
        x1 = 0.0
        y1 = 0.0
        x2 = 0.0
        y2 = 0.0
        print('222   sloper ',sloper)
        if sloper > 0:
            x1 = 0.0
            y1 = 0.0
            yInter = float(listImgHeight[i])
            x2 = float(listImgWidth[i])
            y2 = float(sloper * x2) + yInter
        elif sloper < 0:
            x1 = 0.0
            y1 = 0.0 
            yInter = 0.0
            x2 = float(listImgWidth[i])
        elif sloper == 100:
            x1 = float(listImgWidth[i]) / 2.0
            y1 = 0.0
            x2 = x1
            y2 = float(sloper * x2)
        else:
            x1 = 0.0
            y1 = 0.0
            x2 = float(listImgWidth[i]) / 2.0
            y2 = float(sloper * x2) + yInter
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        
        cv.line(listImages[i], (x1, y1), (x2, y2), listColor[i], 2)
        
    cv.imwrite('C:\\Users\\jesse\\Google Drive\\School\\usf\\ComputerVision\\Group\\images\\Display\\slopeclusters\\slopes' + str(i) + '.png', listImages[i]) 

# write hough line images
for h in range(len(listEuclidPointsAllImgs)):
    for i in range(len(listEuclidPointsAllImgs[h])):
        x1 = listEuclidPointsAllImgs[h][i][0]
        y1 = listEuclidPointsAllImgs[h][i][1]
        x2 = listEuclidPointsAllImgs[h][i][2]
        y2 = listEuclidPointsAllImgs[h][i][3]
        
        
        cv.line(listImages[h], (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    #cv.imshow('slopes', listImages[h])
    cv.imwrite('C:\\Users\\jesse\\Google Drive\\School\\usf\\ComputerVision\\Group\\images\\Display\\hough\\houghLines' + str(h) + '.png', listImages[h]) 

    



'''

redImagePixels = []
reducedKOutput = []
redKOutSlope = []
reduceInc = 40
# create reduced image points 
for i in range(len(listEuclidPointsAllImgs)):
    for j in range(0, len(listEuclidPointsAllImgs[i]), reduceInc):
        redImagePixels.append(listEuclidPointsAllImgs[i][j])
        
# create reduced kOutput points 
for i in range(0, len(kOutput), reduceInc):
    reducedKOutput.append(kOutput[i])
    

for i in range(0, len(kOutputSlope), reduceInc):
   redKOutSlope.append(kOutputSlope[i])
    
    
print('222   ....  len(redImagePixels) ', len(redImagePixels))
print('222   ....  len(reducedKOutput) ', len(reducedKOutput))
print('222   ....  len(redKOutSlope) ', len(redKOutSlope))




#plt.show()
#cv.imshow('image', listImages[0])

# generate color list
listColor = [];
inc = 255.0 / numOfClusters
rgbDivide = int(255.0 / (numOfClusters * 3.0))
col = 0
isTopColor = True
for i in range(numOfClusters):
    if (isTopColor):
        if i < rgbDivide:
            col = (int(i * inc), 0, 255 - int(i * inc))
        elif i >= rgbDivide and i < rgbDivide * 2:
            col = (0, int(i * inc), 0)
        elif i >= rgbDivide * 2 and i < rgbDivide * 3:
            col = (int(i * inc), 0, 0)
        
        isTopColor = False
    else:
        if i < rgbDivide:
            col = (0, int(i * inc), 0)
        elif i >= rgbDivide and i < rgbDivide * 2:
            col = (int(i * inc),  int(i * inc), 255 - int(i * inc))
        elif i >= rgbDivide * 2 and i < rgbDivide * 3:
            col = (255 - int(i * inc), 0, 0)
        isTopColor = True
    
    
    listColor.append(col)
    
print('333     9999         len  listColor ', len(listColor))

print('333     9999           listColor ', listColor)

incScatter = float(len(redImagePixels)) / float(numOfClusters)
colorTup = (0, 0, 0)

for i in range(len(redImagePixels)):
    #print('3333     i ', i)
    #print('3333     redImagePixels ', redImagePixels[i])
    x1 = redImagePixels[i][0]
    y1 = redImagePixels[i][1]
    

    colorIndex = kOutputSlope[i]
    colorTup = listColor[colorIndex]
    
    rhex = hex(colorTup[0])
    ghex = hex(colorTup[1])
    bhex = hex(colorTup[2])
    
    rhex = rhex[2:]
    ghex = ghex[2:]
    bhex = bhex[2:]    
    
    
    if len(rhex) == 1:
        rhex = '0' + rhex
    if len(ghex) == 1:
        ghex = '0' + ghex
    if len(bhex) == 1:
        bhex = '0' + bhex
        
    colorhex = '#' + rhex + ghex + bhex

    #print('333  r hex ', rhex)
    #print('333  ghex ', ghex)
    #print('333  bhex ', bhex)
    #print('333  colorhex ', colorhex)

    
    #plt.scatter(x1, y1, color = colorhex)
    

        
#plt.show()




print('444  ^^^^   len kOutput  ', len(kOutputSlope))
listCountK = []
for i in range(numOfClusters):
    listCountK.append(0)
    for j in range(len(kOutputSlope)):
        if kOutputSlope[j] == i:
            listCountK[i] += 1
            

numOfTopClusters = 5
maxCount = -1
countIt = 0
maxIndex = -1
listMaxCount = []
listMaxIndex = []

for i in range(numOfTopClusters):
    for j in range(len(listCountK)):
        if j == 0:
            maxCount = listCountK[j]
            maxIndex = j
        else:
            print('444  !!!! listMaxIndex  ', listMaxIndex)

            if listCountK[j] > maxCount:
                if j not in listMaxIndex:
                    maxCount = listCountK[j]
                    maxIndex = j
                    print('444  !!!! listCountK[j] > maxCount   maxCount  ', maxCount)

                
    listMaxCount.append(maxCount)
    listMaxIndex.append(maxIndex)
    #listCountK = np.delete(listCountK, maxIndex)
    maxCount = -1
    maxIndex = -1


print('444  ^^^^     len(listMaxCount)  ', len(listMaxCount))
print('444  ^^^^     len(listMaxIndex)  ', len(listMaxIndex))
print('444  ^^^^   listMaxCount)  ', listMaxCount)
print('444  ^^^^   listMaxIndex)  ', listMaxIndex)
print('444  ^^^^   listCountK len  ', len(listCountK))
print('444  ^^^^   listCountK  ', listCountK)

                
# Get slopes of top  clusters
# kOutput contains the point indices for clusters  -> kInput

listSlopes = []
print('555  @@@   kInput len  ', len(kInput))
print('555  @@@   kOutput len  ', len(kOutput))
for i in range(numOfClusters):
    listSlopes.append([])
print('555  @@@   listSlopes len  ', len(listSlopes))

for i in range(0, len(kInput), 2):
    if i < 5:
        print('555  @@@   kInput[', i, '] ', kInput[i])
        print('555  @@@   kOutput[', i, '] ', kOutput[i])
        
    x1 = kInput[i][0]
    y1 = kInput[i][1]
    x2 = kInput[i + 1][0]
    y2 = kInput[i + 1][1]
    clu = kOutput[i]
    
    diffY = y2 - y1
    diffX = x2 - x1
    if (diffX > 0):
        slope = float(diffY) / float(diffX)
    else:
        slope = 99999
        
    listSlopes[clu].append(slope)
    
    if i < 5:
        print('555  @@@   x1 ', x1)
        print('555  @@@   y1 ', y1)
        print('555  @@@   x2 ', x2)
        print('555  @@@   y2 ', y2)
        print('555  @@@   clu ', clu)
        print('555  @@@   diffY ', diffY)
        print('555  @@@   diffX ', diffX)
        print('555  @@@   slope ', slope)


# Find average of slopes in top clusters

slopeSum = 0.0
slopeAvg = 0.0
listAvgSlopeByClusterIndex = []
#print('666  ***  listMaxIndex[', i, ']  ', listMaxIndex[i])
print('666  ***        len   listSlopes  ', len(listSlopes))
for i in range(numOfTopClusters):
    print('666  ***  listMaxIndex[', i, ']  ', listMaxIndex[i])
    slopeSum = 0.0
    for j in range(len(listSlopes[i])):
        slopeSum += listSlopes[i][listMaxIndex[i]]
    if float(len(listSlopes[i])) != 0:
        slopeAvg = slopeSum / float(len(listSlopes[i]))
        listAvgSlopeByClusterIndex.append(slopeAvg)
    else:
        listAvgSlopeByClusterIndex.append(100.0)
    
print('666  ***   listAvgSlopeByClusterIndex len  ', len(listAvgSlopeByClusterIndex))
print('666  ***   listAvgSlopeByClusterIndex  ', listAvgSlopeByClusterIndex)
print('666  ***   listEuclidPointsAllImgs len ', len(listEuclidPointsAllImgs))
print('666  ***   listEuclidPointsAllImgs[0] len ', len(listEuclidPointsAllImgs[0]))



# display lines on image

for i in range(numOfTopClusters):
    slopeTop = listAvgSlopeByClusterIndex[i]
    
    yInter = 0.0
    x1 = 0.0
    x2 = 0.0
    if slopeTop > 0:
        x1 = 0.0
        y1 = float(listImgHeight[0])
        yInter = float(listImgHeight[0])
        x2 = float(listImgWidth[0])
        y2 = float(slopeTop * x2) + yInter
    elif slopeTop < 0:
        x1 = 0.0
        y1 = 0.0 
        yInter = 0.0
        x2 = float(listImgWidth[0])
    elif slopeTop == 100:
        x1 = float(listImgWidth[0]) / 2.0
        y1 = 0.0
        x2 = x1
        y2 = float(slopeTop * x2)
    else:
        print('666 &&&& HORIZ')

    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    
    print('666  ***     i ', i)
    print('666  ***     x1 ', x1)
    print('666  ***     y1 ', y1)
    print('666  ***     x2 ', x2)
    print('666  ***     y2 ', y2)
    print('666  ***     slopeTop ', slopeTop)

    
    #cv.line(listImages[0],(x1,y1),(x2,y2),(0,0,255),1)

#cv.imshow('slopes', listImages[0])


# show image with lines all the way across image

'''



# display
'''
for i in range(len(listEuclidPointsAllImgs)):

    for j in range(len(listEuclidPointsAllImgs[i])):
       # print('999  groupys    listGroupLinesAllImgs[', i, '][', j, '] ', listGroupLinesAllImgs[i][j])

        x1 = listEuclidPointsAllImgs[i][j][0]
        y1 = listEuclidPointsAllImgs[i][j][1]
        x2 = listEuclidPointsAllImgs[i][j][2]
        y2 = listEuclidPointsAllImgs[i][j][3]
        if i == 0:
            plt.scatter(x1, y1) 
            plt.scatter(x2, y2)
            cv.line(listImages[i],(x1,y1),(x2,y2),(0,0,255),1)
        #cv.line(listImages[i],(x1,y1),(x2,y2),(0,0,255),1)
    #cv.imwrite('hougherFewer' + str(i) + '.png', listImages[i]) 
       # plt.plot(x1, y1)
        #plt.plot(x2, y2, 'ro')
    #plt.show()
    
#plt.scatter(2, 3)    
#plt.show()
#cv.imshow('image', listImages[0])
'''







'''

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
'''




    
    
    
    
    
    

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