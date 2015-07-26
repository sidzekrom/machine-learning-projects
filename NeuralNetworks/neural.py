import cv2
import numpy as np

train_image = cv2.imread('images/in1.png', 0)
#crop the image for consistency
h1 = 0
h2 = 300
l1 = 0
l2 = 700
H1 = 0
L1 = 0
H2, L2 = train_image.shape[:2]
img_in = train_image[h1:h2, l1:l2]
img_out = train_image
#make it into a long array
start = np.reshape(img_in, (1, (l2-l1)*(h2-h1)))
end = np.reshape(img_out, (1, (L2-L1)*(H2-H1)))
#Begin neural networks now
#initialize layer dimensions and transition weights
dimensions = [start.shape[1], 100, end.shape[1]]
#dimensions = [10, 5, 10]
transitions = [np.random.randn(dimensions[x], dimensions[x+1]) for x in range(0, len(dimensions)-1)]
print("Initialized transitions")
def get_output():
    S = start
    for x in range(0, len(transitions)):
        S = np.dot(S, transitions[x])
        print S
        S = np.tanh(S)
    return S

out = get_output()

out = np.reshape(out, (H2-H1, L2-L1))
print(out)
#cv2.imshow('image', img_in)
#cv2.imshow('image2', img_out)
#cv2.imshow('output', out)
#cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite('out.png', img_in)
