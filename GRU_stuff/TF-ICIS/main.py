import os
import pickle
from GRU import *
import matplotlib.pyplot as plt

def main():
	subsetNum = 25
	M = 50
	N = 10

	"""
		Creating the CNN variables

	"""
	# TODO: create the CNN variables

	"""
		Creating the GRU cell variables

	"""
	Z_Stack = list()
	ZT_Stack = list()
	R_Stack = list()
	RT_Stack = list()
	P_Stack = list()
	PT_Stack = list()
	H_Stack = list()

	# W matricies are multiplied with h(t-1)
	Wz = tf.get_variable(name = "Wz", shape = (N, M), initializer = tf.truncated_normal_initializer(stddev = 0.1))
	Wr = tf.get_variable(name = "Wr", shape = (N, M), initializer = tf.truncated_normal_initializer(stddev = 0.1))
	Wp = tf.get_variable(name = "Wp", shape = (N, M), initializer = tf.truncated_normal_initializer(stddev = 0.1))

	# U matricies are multiplied with x(t-1)
	Uz = tf.get_variable(name = "Uz", shape = (N, N), initializer = tf.truncated_normal_initializer(stddev = 0.1))
	Ur = tf.get_variable(name = "Ur", shape = (N, N), initializer = tf.truncated_normal_initializer(stddev = 0.1))
	Up = tf.get_variable(name = "Up", shape = (N, N), initializer = tf.truncated_normal_initializer(stddev = 0.1))
	
	# b vectors are N dimensional biases
	bz = tf.get_variable(name = "bz", shape = (N, 1), initializer = tf.truncated_normal_initializer(stddev = 0.1))
	br = tf.get_variable(name = "br", shape = (N, 1), initializer = tf.truncated_normal_initializer(stddev = 0.1))
	bp = tf.get_variable(name = "bp", shape = (N, 1), initializer = tf.truncated_normal_initializer(stddev = 0.1))

	# state h starts as a constant of 0s
	h = tf.zeros(name = "h", shape = (N, 1), dtype = tf.float32)

	# x is a placeholder for each of the CNN feature maps
	x = list()
	for i in range(subsetNum):
		x.append(
			tf.placeholder(
				tf.float32, 
				shape = [M, 1],
				name = "FeatureSpace_" + str(i)
			)
		)

	"""
		Defining the network

	"""
	for i in range(subsetNum):
		h = GRU_Cell(
			M, N, h, x[i], 	# External Variables
			Wz, Wr, Wp, 	# Weight Matricies for x
			Uz, Ur, Up, 	# Weight Matricies for h
			bz, br, bp, 	# biases for FC layers
			Z_Stack, ZT_Stack, R_Stack, RT_Stack, P_Stack, PT_Stack, H_Stack
		)

	init_op = tf.global_variables_initializer()

	d = dict()
	for i in range(subsetNum):
		d[x[i]] = np.random.rand(M, 1)

	with tf.Session() as sess:
		sess.run(init_op)

		"""
			Command to view tensorboard:
			cd C:\\Users\\dmorl\\Desktop\\File_Folder\\coding\\Python\\Computer Vision\\Tensorflow\\TF-ICIS\\graphs\\
			tensorboard --logdir=.\\
			
		"""
		writer = tf.summary.FileWriter('graphs', sess.graph)
		print(sess.run(h, feed_dict = d))

	return

if __name__ == "__main__":
	main()