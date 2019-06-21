import math
import numpy as np
from numba import cuda

@cuda.jit
def runCNNKernel(image, kernels, poolInfo, ordering):
	pass

def runCNN(image, kernels, poolInfo, ordering):
	return image;