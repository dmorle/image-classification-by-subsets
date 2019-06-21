import math
import numpy as np
from numba import cuda

@cuda.jit
def maxpoolKernel(image, size, stride, output):
	x, y, z = cuda.grid(3)
	outShape = output.shape
	
	if x < outShape[0] and y < outShape[1] and z < outShape[2]:
		# do shit
		x_s = x * stride[0]
		y_s = y * stride[1]
		currMax = 0
		for i in range(size[0]):
			for j in range(size[1]):
				if currMax < image[x_s + i][y_s + j][z]:
					currMax = image[x_s + i][y_s + j][z]

		output[x, y, z] = currMax
	return

def maxpool(image, size = np.array((2, 2)), stride = np.array((2, 2)), output = None):
	if not output:
		output = np.empty([(image.shape[i]-size[i])//stride[i] +  1 for i in range(2)] + [image.shape[2]])
	
	# checking for errors
	for i in [0, 1]:
		if (image.shape[i]-size[i])%stride[i]:
			raise(Exception("invalid pool layout"))

	# setting up the kernel
	threadsperblock = (8, 8, 8)
	blockspergrid = tuple((math.ceil(output.shape[i] / threadsperblock[i]) for i in range(3)))

	maxpoolKernel[threadsperblock, blockspergrid](image, size, stride, output)
	return output

def main():
	image = np.random.rand(100, 100, 3)	# x, y, z
	size = np.array((5, 5))				# x, y
	stride = np.array((5, 5))

	output = maxpool(image, size, stride)

	print(output)

	return

if __name__ == '__main__':
	main()