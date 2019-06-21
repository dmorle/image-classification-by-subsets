import math
import numpy as np
from numba import cuda

@cuda.jit
def convolutionKernel(image, kernels, stride, output):
	# this is the same as x, y, kernelNum
	x, y, z = cuda.grid(3)
	outShape = output.shape
	
	if x < outShape[0] and y < outShape[1] and z < outShape[2]:

		# set up the bounds for the convolution
		kernelShape = kernels.shape

		# apply convolution
		pixelVal = 0.0
		for i in range(kernelShape[0]):
			for j in range(kernelShape[1]):
				for k in range(kernelShape[2]):
					pixelVal += image[i + x*stride][j + y*stride][k] * kernels[i][j][k][z]
		
		# apply ReLU
		if pixelVal < 0:
			pixelVal = 0

		output[x][y][z] = pixelVal

	return

def convolution(image, kernels, stride = np.array((1, 1)), output = None):
	if not output:
		output = np.zeros([(image.shape[i]-kernels.shape[i])//stride +  1 for i in range(2)] + [kernels.shape[3]])	# x, y, z ;; z comes from kernelNum

	# checking for errors
	for i in [0, 1]:
		if (image.shape[i]-kernels.shape[i])%stride[i]:
			raise(Exception("invalid shape"))

	if kernels[2] != image[2]:
		raise(Exception("invalid shape"))

	# setting up the kernel
	threadsperblock = (8, 8, 8)
	blockspergrid = tuple((math.ceil(output.shape[i] / threadsperblock[i]) for i in range(3)))

	convolutionKernel[threadsperblock, blockspergrid](image, kernels, stride, output)
	return output

def main():
	image = np.random.rand(100, 100, 3)			# x, y, z
	kernels = np.random.rand(10, 10, 3, 50)		# x, y, z, kernelNum
	
	output = convolution(image, kernels)

	print(output)

	return

if __name__ == '__main__':
	main()