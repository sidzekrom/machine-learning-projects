##this is just code taken from the images project
#and modified to fit this new input

import numpy as np
from random import randint
data = []
f = open('agaricus-lepiota.data.txt')

for row in f:
    line = row.split(",")
    line[len(line)-1] = line[len(line)-1][0]
    data.append(line)

data = np.array(data)
data = np.vectorize(lambda x: ord(x))(data)
nu=0.0000001 #our value of nu
(height, width) = np.shape(data);
print(np.shape(data))
#Begin neural networks now
#initialize layer dimensions and transition weights
np.random.seed(5853451)
dimensions = [width-1, 1]
#dimensions = [10, 5, 10]
#transitions = [(2.5/dimensions[0]) * (np.random.rand(dimensions[0], dimensions[1])),\
#                (2.5/dimensions[1]) * (np.random.rand(dimensions[1], dimensions[2]))]
transitions = [(2.5/dimensions[0]) * (np.random.rand(dimensions[0], dimensions[1]))]

def back_propagation(in_img, out_img):
    S = in_img
    intermediate_values = [S]
    for x in range(0, len(transitions)):
        S = np.dot(S, transitions[x])
        S = np.tanh(S)
        intermediate_values.append(S)
    #deltas are the way that the error function changes as a specific node changes. It is in reverse order
    deltas = []
    #we use RMS error. Therefore, this is the last delta value
    deltas.append(2*(intermediate_values[-1]-out_img))
    #Computing the previous deltas. I believe that the previous deltas should be the next deltas,
    #times (the transpose of the transition matrices with each row scaled up by 1-value^2 of the corresponding
    #next neuron's value, which is actually stored in intermediate_values!)   
    for x in range(len(intermediate_values)-2, -1, -1):
        weights    = transitions[x].copy().transpose()
        operations = intermediate_values[x+1]
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
        for y in range(0, len(intermediate_values[x])):
            transitions[x][y]-=(nu*intermediate_values[x][y])*deltas[x+1][0]

def neural_net():
    #This neural net function will use stochastic gradient descent and weight elimination
    for x in range(0, 5000):
        if(x % 1000 == 0):
            print(x)
        randY = randint(1, height/2)
        in_img = data[randY,1:width]
        out_img = data[randY,0:1]
        print(transitions)
        back_propagation(in_img, out_img)
    count = 0
    for x in range(height/2+1, height):
        z = get_output(data[x,1:width])
        if(abs(ord('p') - z) < abs(ord('e') - z)):
            #classified as 'p'
            if(data[x][0] != ord('p')):
                count+=1
        else:
            if(data[x][0] != ord('e')):
                count+=1
    total = height - height/2
    print(float(count)/total)

def backProp(layersin, current):
    #layersin = [start]
    #current = start
    transLen = len(transitions)
    deltastream = []
    for i in range(transLen):
        current = np.dot(current, transitions[i])
        current = np.tanh(current)
        layersin.append(current)
    #length of layersin is len(transitions) + 1
    #maintain a series of delta for each output neuron
    errorgrad = []
    desiredoutput = end[0]
    actualoutput = layersin[transLen][0]
    deltaseed = 1 - ((actualoutput) ** 2)
    for i in range(len(end[0])):    
        currentdelta = np.array([1 - ((actualoutput[i]) ** 2)])
        deltacollection = [currentdelta]
        for j in range(transLen-1):
            if (j == 0):
                derivative = (1 - layersin[transLen-1] * layersin[transLen-1])
                update = np.dot(np.matrix(currentdelta),\
                    np.matrix((transitions[transLen-1].T)[i]))
                currentdelta = derivative * np.array(update)
            else:
                derivative = 1 - layersin[transLen-i-1] *\
                    layersin[transLen-i-1]
                update = np.dot(np.matrix(currentdelta),\
                    np.matrix(transitions[transLen-i-1].T))
                currentdelta = derivative * np.array(update)
            deltacollection.append(currentdelta)
        deltacollection.reverse()
        deltastream.append(deltacollection)
    for i in range(transLen):
        #this iteration we will compute the error derivative with respect to
        #the i-th weight matrix
        errorgradterm = 0
        ithoutput = layersin[i].T
        if (i == transLen-1):
            errorgradterm = np.matrix([deltaseed]) *\
                    (actualoutput[j] - desiredoutput[j])
        else:    
            for j in range(len(end[0])):
                errorgradterm += deltastream[j][i] *\
                         (actualoutput[j] - desiredoutput[j])
        #now errorgradterm is the sum of all deltas
        errorgradterm = np.dot(ithoutput, errorgradterm)
        errorgrad.append(errorgradterm)
    #length of errorgrad is the same as length of transitions.
    #errorgrad[i] is gradient of error with respect to weight matrix i
    for i in range(transLen):
        #print transitions[i].shape
        #print errorgrad[i].shape
        transitions[i] -= nu * errorgrad[i]


def get_output(test):
    #once we have trained the weights, call get_output on a test image
    S = test
    for x in range(0, len(transitions)):
        S = np.dot(S, transitions[x])
        S = np.tanh(S)
    return S


neural_net()

