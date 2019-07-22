import tensorflow as tf
import numpy as np

# returns an operation representing a fully connected layer
def FCLayer(X, W, b = None, activation = None, name = None):
	if not name:
		name = "FCLayer"
	with tf.name_scope(name):
		if b:
			XW = tf.matmul(X, W, name = "RawProd")
			if activation:
				Z = tf.add(XW, b, name = "Z")
				return activation(Z, name = "A")
			Z = tf.add(XW, b, name = "A")
			return Z
		if activation:
			XW = tf.matmul(X, W, name = "RawProd")
			return activation(XW, name = "A")
		return tf.matmul(X, W, name = "A")

# pre-defined convolutional neural network structure
# X is a batch of image subsets
# K is a list of kernels
def convNet(X, K, batSize, name = None):
	if not name:
		name = "CNN"
	with tf.name_scope(name):
		with tf.name_scope("Conv1"):
			X = tf.nn.convolution(X, K[0], padding = "VALID", name = "LinearConv")
			X = tf.nn.relu(X)
		with tf.name_scope("Pool1"):
			X = tf.nn.max_pool(X, ksize = (batSize, 2, 2, 1), strides = (1, 2, 2, 1), padding = "VALID", name = "MaxPool")
	return X

def LSTMCell(C, h, x, Wf, Wi, Wc, Wo, bf, bi, bc, bo, name = None):
	if not name:
		name = "LSTMCell"
	with tf.name_scope(name):
		x_h = tf.concat((x, h), 1)
		f = FCLayer(x_h, Wf, bf, tf.nn.sigmoid, name = name + "_f")
		i = FCLayer(x_h, Wi, bi, tf.nn.sigmoid, name = name + "_i")
		c = FCLayer(x_h, Wc, bc, tf.nn.tanh   , name = name + "_c")
		o = FCLayer(x_h, Wo, bo, tf.nn.sigmoid, name = name + "_o")

		C_f = tf.multiply(f, C)
		i_c = tf.multiply(i, c)
		C = tf.add(C_f, i_c)

		C_tanh = tf.nn.tanh(C)
		h = tf.multiply(C_tanh, o)

		return (C, h)

# pre-defined network containing the structure for the entire network
def createNetwork(Inputs, batSize, subsetNum, cellState, hiddenState, m, n):

	# create the kernels' variables
	K = [
			# 5 kernels size 5x5
			tf.get_variable(name = "Conv1_Kernel", shape = (5, 5, 1, 5), initializer = tf.truncated_normal_initializer(stddev = 0.01))
		]

	# creating the LSTM cell variables
	Wf = tf.get_variable(name = "Wf", shape = (m, n), initializer = tf.truncated_normal_initializer(stddev = 0.01))
	Wi = tf.get_variable(name = "Wi", shape = (m, n), initializer = tf.truncated_normal_initializer(stddev = 0.01))
	Wc = tf.get_variable(name = "Wc", shape = (m, n), initializer = tf.truncated_normal_initializer(stddev = 0.01))
	Wo = tf.get_variable(name = "Wo", shape = (m, n), initializer = tf.truncated_normal_initializer(stddev = 0.01))
	
	bf = tf.get_variable(name = "bf", shape = (1, n), initializer = tf.truncated_normal_initializer(stddev = 0.01))
	bi = tf.get_variable(name = "bi", shape = (1, n), initializer = tf.truncated_normal_initializer(stddev = 0.01))
	bc = tf.get_variable(name = "bc", shape = (1, n), initializer = tf.truncated_normal_initializer(stddev = 0.01))
	bo = tf.get_variable(name = "bo", shape = (1, n), initializer = tf.truncated_normal_initializer(stddev = 0.01))

	# define the CNN operation
	X = list()
	for i in range(subsetNum):
		X.append(convNet(Inputs[i], K, batSize))

	# define the LSTM operation
	for x in X:
		x = tf.reshape(x, [1, -1])
		cellState, hiddenState = LSTMCell(
			cellState, 
			hiddenState, 
			x,
			Wf, Wi, Wc, Wo,
			bf, bi, bc, bo
		)

	return hiddenState

def main():
	# CNN parameters
	hLevel = 0
	imageSize = 28
	batSize = 1

	sSize = 2**(hLevel + 1)
	subsetNum = sSize * sSize

	# creating the input placeholders
	# Inputs is list of (a batch of images) by subsets
	Inputs = list()
	for i in range(subsetNum):
		Inputs.append(
			tf.placeholder(
				tf.float32, 
				shape = [
					batSize, 
					int(imageSize/sSize), 
					int(imageSize/sSize), 
					1
				],
				name = "InputSubset_" + str(i)
			)
		)

	# LSTM parameters
	m = 135
	n = 10

	# Cell and hidden state are contants initialized with 0s
	cellState = tf.zeros(
		shape = [
			1,
			n
		],
		dtype = tf.float32,
		name = "cellState"
	)

	hiddenState = tf.zeros(
		shape = [
			1,
			n
		],
		dtype = tf.float32,
		name = "hiddenState"
	)

	network = createNetwork(Inputs, batSize, subsetNum, cellState, hiddenState, m, n)

	init_op = tf.global_variables_initializer()

	# initialization for inputs
	# eventually, this will involve creating the image subsets, and initializing d[Input[i]] with the ith subset
	d = dict()
	for i in range(subsetNum):
		d[Inputs[i]] = np.random.rand(
			batSize, 
			int(imageSize/sSize), 
			int(imageSize/sSize), 
			1
		)

	# running the network
	with tf.Session() as sess:
		sess.run(init_op)

		writer = tf.summary.FileWriter('graphs', sess.graph)
		print(sess.run(network, feed_dict = d))

	return

if __name__ == "__main__":
	main()