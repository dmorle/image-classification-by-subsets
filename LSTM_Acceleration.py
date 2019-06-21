import math
import numpy as np
from numba import cuda

# vec and output must be np.ndarray with len(shape) = 1
@cuda.jit
def tanhVec_K(vec, output, size = 0):
	x = cuda.grid(1)

	if not size:
		size = vec.shape[0]

	if x < size:
		# compute tanh
	return

# vec and output must be np.ndarray with len(shape) = 1
@cuda.jit
def sigmoidVec_K(vec, output, size = 0):
	x = cuda.grid(1)

	if not size:
		size = vec.shape[0]

	if x < size:
		#compute sigmoid
	return

# mrx and output must be np.ndarray with len(shape) = 2
@cuda.jit
def tanhMrx_K(mrx, output, size = None):
	x, y = cuda.grid(2)

	if not size:
		size = mrx.shape

	if x < size[0] and y < size[1]:
		#compute tanh
	return

@cuda.jit
def sigmoidMrx_K(mrx, output, size = None):
	x, y = cuda.grid(2)

	if not size:
		size = mrx.shape

	if x < size[0] and y <size[1]:
		#compute sigmoid
	return
