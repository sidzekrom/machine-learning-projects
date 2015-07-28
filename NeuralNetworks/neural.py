import cv2
import numpy as np

train_image = cv2.imread('images/in1.png', 0)
#crop the image for consistency
h1 = 0
h2 = 30
l1 = 0
l2 = 70
H1 = 0
L1 = 0
nu=0.1 #our value of nu
H2, L2 = train_image.shape[:2]
train_image = np.vectorize(lambda x: x/256.0)(train_image)
img_in = train_image[h1:h2, l1:l2]
img_out = train_image
#make it into a long array
start = np.reshape(img_in, (1, (l2-l1)*(h2-h1)))
end = np.reshape(img_out, (1, (L2-L1)*(H2-H1)))
#Begin neural networks now
#initialize layer dimensions and transition weights
dimensions = [start.shape[1], 100, end.shape[1]]
#dimensions = [10, 5, 10]
transitions = [(100.0/dimensions[0]) * (2 * np.random.rand(dimensions[0], dimensions[1]) - 1.0),\
                (100.0/dimensions[1]) * (2 * np.random.rand(dimensions[1], dimensions[2]) - 1.0)]

print("Initialized transitions")
def back_propagation():
	S = start
	intermediate_values = [S]
	for x in range(0, len(transitions)):
		S = np.dot(S, transitions[x])
		S = np.tanh(S)
		intermediate_values.append(S)
	#deltas are the way that the error function changes as a specific node changes. It is in reverse order
	deltas = []
	#we use RMS error. Therefore, this is the last delta value
	deltas.append(2*(intermediate_values[-1]-end))
	#Computing the previous deltas. I believe that the previous deltas should be the next deltas,
	#times (the transpose of the transition matrices with each row scaled up by 1-value^2 of the corresponding
	#next neuron's value, which is actually stored in intermediate_values!)
	
	for x in range(len(intermediate_values)-2, -1, -1):

		weights	= transitions[x].copy().transpose()
		operations = intermediate_values[x+1][0]
		#multiply the ith row of weights by the ith element of operations
		#probably is a faster way to do it using np.multiply
		for y in range(0, len(operations)):
			value = 1-operations[y]**2
			weights[y] = (np.vectorize(lambda z:z*value))(weights[y])

		delta_prev = np.dot(deltas[-1], weights)
		deltas.append(delta_prev)
	deltas.reverse()
	#Now, we have the values and the deltas. It is time to update the transitions
	for x in range(0, len(transitions)):
		#take the deltas and intermediate_values and array multiply together
		deltas[x+1]*=intermediate_values[x+1]
		#Once again, there is probably a faster way to do this
		#We take the result and scale it by the previous node's value. Then, we add to the transitions, which is now the updated version
		for y in range(0, len(intermediate_values[x][0])):
			transitions[x][y]+=(nu*intermediate_values[x][0][y])*deltas[x+1][0] #move 0.1 times the gradient at a time
		


def get_output():
    S = start
    for x in range(0, len(transitions)):
        S = np.dot(S, transitions[x])
        S = np.tanh(S)
    return S

back_propagation()
out = get_output()

out = np.reshape(out, (H2-H1, L2-L1))
out = (np.vectorize(lambda x: (x + 1.0) / 2.0))(out)
"""
print("out image")
print(out)
cv2.imshow('image', img_in)
cv2.imshow('image2', img_out)
cv2.imshow('output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite('out.png', img_in)
"""