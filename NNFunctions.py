from numba import cuda
import numpy as np

@cuda.jit
def sigmoidKernel(array, size):
	x = cuda.grid(1)

	if x < size:
		# reference within the array
		power = 
	return

@cuda.jit
def sigmoid1Kernel(array, size):
	return

@cuda.jit
def tanhKernel(array, size):
	return

@cuda.jit
def tanh1Kernel(array, size):
	return


def sigmoid(array):
	size = len(array)
	threadsperblock = 32
	blockspergrid = math.ceil(size / threadsperblock)

	sigmoidKernel[threadsperblock, blockspergrid](array, size)
	return

def sigmoid1(array):
	size = len(array)
	threadsperblock = 32
	blockspergrid = math.ceil(size / threadsperblock)

	sigmoid1Kernel[threadsperblock, blockspergrid](array, size)
	return

def tanh(array):
	size = len(array)
	threadsperblock = 32
	blockspergrid = math.ceil(size / threadsperblock)

	tanhKernel[threadsperblock, blockspergrid](array, size)
	return

def tanh1(array):
	size = len(array)
	threadsperblock = 32
	blockspergrid = math.ceil(size / threadsperblock)

	tanh1Kernel[threadsperblock, blockspergrid](array, size)
	return