import numpy as np
import math

def pythonListTesting():
	return

def numpyArrayTesting():
	shape = (10)
	arr = np.random.random_sample(shape)
	print(arr)
	# print(arr.shape)
	# print(len(arr))
	modifyArray(arr)
	print(arr)
	return

def modifyArray(arr):
	arr[0] = 0
	return

def main():
	#numpyArrayTesting()

	n = 2**31
	e = math.exp(n)
	return

if __name__ == "__main__":
	main()