import os
import tempfile

import time

import cv2
import numpy as np
from PIL import Image

def calcVanishingPoint(lines):
    points = lines[:, :2]
    normals = lines[:, 2:4] - lines[:, :2]
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-4)
    normals = np.stack([normals[:, 1], -normals[:, 0]], axis=1)
    normalPointDot = (normals * points).sum(1)

    if lines.shape[0] == 2:
        VP = np.linalg.solve(normals, normalPointDot)
    else:
        VP = np.linalg.lstsq(normals, normalPointDot)[0]
        pass
    return VP


def calcVanishingPoints(allLines, numVPs):
    distanceThreshold = np.sin(np.deg2rad(5))
    lines = allLines.copy()
    VPs = []
    VPLines = []
    for VPIndex in range(numVPs):
        points = lines[:, :2]
        lengths = np.linalg.norm(lines[:, 2:4] - lines[:, :2], axis=-1)
        normals = lines[:, 2:4] - lines[:, :2]
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-4)
        normals = np.stack([normals[:, 1], -normals[:, 0]], axis=1)
        maxNumInliers = 0
        bestVP = np.zeros(2)
        #for _ in range(int(np.sqrt(lines.shape[0]))):
        for _ in range(min(pow(lines.shape[0], 2), 100)):
            sampledInds = np.random.choice(lines.shape[0], 2)
            if sampledInds[0] == sampledInds[1]:
                continue
            sampledLines = lines[sampledInds]
            try:
                VP = calcVanishingPoint(sampledLines)
            except:
                continue

            inliers = np.abs(((np.expand_dims(VP, 0) - points) * normals).sum(-1)) / np.linalg.norm(np.expand_dims(VP, 0) - points, axis=-1) < distanceThreshold
            
            numInliers = lengths[inliers].sum()
            if numInliers > maxNumInliers:
                maxNumInliers = numInliers
                bestVP = VP
                bestVPInliers = inliers
                pass
            continue
        if maxNumInliers > 0:
            inlierLines = lines[bestVPInliers]
            VP = calcVanishingPoint(inlierLines)
            VPs.append(VP)
            #print(bestVP)
            #print(inlierLines)
            #print(VP)
            #exit(1)
            VPLines.append(inlierLines)
            lines = lines[np.logical_not(bestVPInliers)]
            pass
        continue
    VPs = np.stack(VPs, axis=0)
    return VPs, VPLines, lines


def estimateFocalLength(image):
    from pylsd.lsd import lsd
    
    height = image.shape[0]
    width = image.shape[1]

    lines = lsd(image.mean(2))

    lineImage = image.copy()
    for line in lines:
        cv2.line(lineImage, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 0, 255), int(np.ceil(line[4] / 2)))
        continue
    #cv2.imwrite('test/lines.png', lineImage)

    numVPs = 3
    VPs, VPLines, remainingLines = calcVanishingPoints(lines, numVPs=numVPs)
    #focalLength = (np.sqrt(np.linalg.norm(np.cross(VPs[0], VPs[1]))) + np.sqrt(np.linalg.norm(np.cross(VPs[0], VPs[2]))) + np.sqrt(np.linalg.norm(np.cross(VPs[1], VPs[2])))) / 3
    focalLength = (np.sqrt(np.abs(np.dot(VPs[0], VPs[1]))) + np.sqrt(np.abs(np.dot(VPs[0], VPs[2]))) + np.sqrt(np.abs(np.dot(VPs[1], VPs[2])))) / 3
    return focalLength


def PlaneDepthLayer(planes, ranges):
  batchSize = 1
  if len(planes.shape) == 3:
    batchSize = planes.shape[0]
    planes = planes.reshape(planes.shape[0] * planes.shape[1], planes.shape[2])
    pass
  
  planesD = np.linalg.norm(planes, 2, 1)
  planesD = np.maximum(planesD, 1e-4)
  planesNormal = -planes / planesD.reshape(-1, 1).repeat(3, 1)

  #print(planesD, planesNormal)
  #print(ranges.min(), ranges.max())
  normalXYZ = np.dot(ranges, planesNormal.transpose())
  normalXYZ[normalXYZ == 0] = 1e-4
  normalXYZ = 1 / normalXYZ
  #print(normalXYZ.min(), normalXYZ.max())
  depths = -normalXYZ
  depths[:, :] *= planesD
  if batchSize > 1:
    depths = depths.reshape(depths.shape[0], depths.shape[1], batchSize, -1).transpose([2, 0, 1, 3])
    pass
  depths[(depths < 0) + (depths > 10)] = 10
  return depths


def calcPlaneDepths(planes, width, height, info):
    urange = np.arange(width, dtype=np.float32).reshape(1, -1).repeat(height, 0) / (width + 1) * (info[16] + 1) - info[2]
    vrange = np.arange(height, dtype=np.float32).reshape(-1, 1).repeat(width, 1) / (height + 1) * (info[17] + 1) - info[6]
    ranges = np.array([urange / info[0], np.ones(urange.shape), -vrange / info[5]]).transpose([1, 2, 0])
    planeDepths = PlaneDepthLayer(planes, ranges)
    return planeDepths

def drawDepthImage(depth):
    #return cv2.applyColorMap(np.clip(depth / 10 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    return 255 - np.clip(depth / 5 * 255, 0, 255).astype(np.uint8)


class ColorPalette:
    def __init__(self, numColors):
        #np.random.seed(2)
        #self.colorMap = np.random.randint(255, size = (numColors, 3))
        #self.colorMap[0] = 0

        
        self.colorMap = np.array([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [255, 230, 180],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [100, 0, 0],
                                  [0, 100, 0],                                   
                                  [255, 255, 0],                                  
                                  [50, 150, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [128, 128, 80],
                                  [0, 50, 128],                                  
                                  [0, 100, 100],
                                  [0, 255, 128],                                  
                                  [0, 128, 255],
                                  [255, 0, 128],                                  
                                  [128, 0, 255],
                                  [255, 128, 0],                                  
                                  [128, 255, 0],                                                                    
        ])

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.random.randint(255, size = (numColors, 3))
            pass
        
        return

    def getColorMap(self):
        return self.colorMap
    
    def getColor(self, index):
        if index >= colorMap.shape[0]:
            return np.random.randint(255, size = (3))
        else:
            return self.colorMap[index]
            pass

def drawSegmentationImage(segmentations, randomColor=None, numColors=22, blackIndex=-1):
    if segmentations.ndim == 2:
        numColors = max(numColors, segmentations.max() + 2, blackIndex + 1)
    else:
        numColors = max(numColors, segmentations.shape[2] + 2, blackIndex + 1)
        pass
    randomColor = ColorPalette(numColors).getColorMap()
    if blackIndex >= 0:
        randomColor[blackIndex] = 0
        pass
    width = segmentations.shape[1]
    height = segmentations.shape[0]
    if segmentations.ndim == 3:
        #segmentation = (np.argmax(segmentations, 2) + 1) * (np.max(segmentations, 2) > 0.5)
        segmentation = np.argmax(segmentations, 2)
    else:
        segmentation = segmentations
        pass
    segmentation = segmentation.astype(np.int)
    return randomColor[segmentation.reshape(-1)].reshape((height, width, 3))