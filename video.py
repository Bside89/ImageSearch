import cv2
import os
from os import path

folder = 'video1'

if not (path.exists(folder)):
    os.mkdir(folder)


vidcap = cv2.VideoCapture('Set01_video01.h264')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("video1/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: %d'%(count))
  count += 1
