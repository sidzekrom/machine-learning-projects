import cv2
import numpy as np

img_in = cv2.imread('images/in1.png', 1)
img_out = cv2.imread('images/out1.png', 1)
#crop the image for consistency
img_in = img_in[0:400, 0:700]
img_out = img_out[0:400, 0:700]
#make it into a long array
start = img_in.flatten()
#Begin neural networks now

dimensions = [len(start), len(start), len(start), len(start)]
transitions = []

"""
cv2.imshow('image', img_in)
cv2.imshow('image2', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
# cv2.imwrite('out.png', img_in)
