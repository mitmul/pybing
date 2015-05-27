#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('build')

import cv2 as cv
import numpy as np
import bing

binger = bing.BING('build/ObjectnessTrainedModel', 2, 8, 2)

img = cv.imread('sample.jpg')
canvas = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

bbox = binger.objectness(img)
for b in bbox:
    x1, y1, x2, y2 = [int(a) for a in b[:4]]
    s = b[-1]
    canvas[y1:y2, x1:x2] += s

canvas /= np.max(canvas)
cv.imwrite('sample_result.jpg', canvas * 255)
