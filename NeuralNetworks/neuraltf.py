import tensorflow
import random

# code for backpropagation on classic neural net
# backpropagation is done in mini-batch style
class NeuralNet:
    def __init__(self, dim, X, Y, nu, initial_weight):
        self.session = tf.InteractiveSession()

        self.dimensions = dim
        
        self.train_data = X
        self.train_labels = Y
        self.nu = nu #learning rate
        self.num_trainexamples = len(train_data)

        self.pointlen = len(X[0])
        self.outputlen = len(Y[0])

        inshape = [None, self.pointlen]
        outshape = [None, self.outputlen]

        self.train_placeholder = tf.placeholder(tf.float32, shape=inshape)
        self.output_placeholder = tf.placeholder(tf.float32, shape=outshape)

        self.transitions = []
        for i in range(len(self.dimensions)-1):
            self.transitions.append(tf.Variable(\
                tf.random_uniform([self.dimensions[i+1], self.dimensions[i]],\
                minval = 0.1, maxval = 1.1)))

        self.init_vars = tf.initialize_all_variables()
        self.outputs = []

        self.outputs.append(self.train_placeholder)

        # compute the final output. This list will be useful for
        # backpropagation later
        for i in range(len(self.dimensions)-1):
            nxt_out = tf.matmul(self.outputs[i], self.transitions[i])
            self.outputs.append(tf.nn.sigmoid(nxt_out))

        # the final output of the neural net
        self.final_output = outputs[len(self.dimensions)-1]

        # using euclidean distance between predictions and actual label
        # as error
        self.euc_error = tf.reduce_mean(\
            tf.reduce_sum((self.final_output - self.output_placeholder)**2,\
            reduction_indices=[1]))

        self.train_step = tf.train.GradientDescentOptimizer(\
            self.nu).minimize(self.euc_error)

    def train(self, num_iterations, batch_size=1):
        for i in range(num_iterations):
            batch = [[],[]]
            for j in range(batch_size):
                random_num = random.randint(0, self.num_trainexamples)
                batch[0].append(self.train_data[random_num])
                batch[1].append(self.train_labels[random_num])

            batch = [tf.convert_to_tensor(batch[0]),\
                tf.convert_to_tensor(batch[1])]

            feed_dict_in = {self.train_placeholder: batch[0],\
                self.output_placeholder: batch[1]}
            
            self.train_step.run(feed_dict=feed_dict_in)

