import random
import pickle
import numpy as np
from Convolution import convolution as conv
from Maxpool import maxpool as maxp

"""
#----------------------------------- this section defines the structure of the CNN net ----------------------------------#

The kLayerSize input is a list of tuples with entries as follows:
	[
	(x, y, z, kernelNum),	# 1st conv layer kernel shape
	(x, y, z, kernelNum),	# 2nd conv layer kernel shape
	:						
	# number of conv layers = len of list
	]

The kStrideSize input is a list of np.ndarray s in form:
	[
	(stride array),	# 1st conv layer stride values
	(stride array),	# 2nd conv layer stride values
	:
	# number of conv layers = len of list
	]

#####################################################################

the poolInfo input is a list of np.ndarray s in form:
	[
	(size tuple), (stride tuple),	# 1st pool layer; type = int32
	(size tuple), (stride tuple),	# 2nd pool layer; type = int32
	:
	# number of pool layers = len of list
	]

#####################################################################

The ordering input is a list of booleans:
	false 	=> convolution layer
	true 	=> maxpool layer

This imposes limitations, somme of which are listed below:
	len(ordering) = len(kLayerSize) + len(poolInfo)

#----------------------------------- this section defines the structure of the LSTM net ----------------------------------#

TBD

"""

class NeuralNetwork:
	def __init__(self, kLayerSize, kStrideSize, poolInfo, ordering, n = 257, p = 100):

		#stuff for the CNN
		self.__kVals__ = list()
		for layerShape in kLayerSize:
			self.__kVals__.append(np.random.random_sample(layerShape))
		self.__kStride__ = kStrideSize
		self.__poolInfo__ = poolInfo
		self.__ordering__ = ordering

		#declaring the LSTM and initializing it with random weights and biases
		self.__n__ = n
		self.__p__ = p
		self.__m__ = n + p

		self.__Wf__, self.__Wi__, self.__Wc__, self.__Wo__ = (np.random.rand(n, self.__m__) for i in range(4))
		self.__bf__, self.__bi__, self.__bc__, self.__bo__ = (np.random.rand(n) for i in range(4))
		return

	def run(self, Image):
		cellState = np.zeros(self.__n__)
		hiddenState = np.zeros(self.__n__)
		for cell in Image:
			# run ConvNet on cell
			convCount = 0
			poolCount = 0
			for layerType in self.__order__:
				if layerType:
					# run pool
					cell = pool(cell, size = self.__poolInfo__[poolCount][0], stride = self.__poolInfo__[poolCount][1])S
					poolCount += 1
				else:
					# run conv
					cell = conv(cell, self.__kVals__[convCount], stride = self.__kStride__[convCount])
					convCount += 1

			# run LSTM
			
			# FML... I dont want to do this.................
		return

	def run(self, Image):
		percentComplete = 0
		C = vector([0]*self.n)
		h = vector([0]*self.n)
		for cellNum in range(len(Image)):
			# running the CNN
			A = Image[cellNum]
			ls = vector([len(Image[cellNum]), len(Image[cellNum][0])]) # x, y
			for gamma in range(len(self.CNNstruct)):
				if self.order[gamma]:
					K = self.kVals[gamma]
					layerInfo = self.CNNstruct[gamma]
					ls -= vector([layerInfo[1], layerInfo[2]]) - vector([1, 1])

					Aprime = list()
					for x0 in range(ls.data[0]):
						row = list()
						for y0 in range(ls.data[1]):
							pixel = list()
							for n0 in range(layerInfo[0]):
								convSum = 0
								for i in range(layerInfo[1]):
									for j in range(layerInfo[2]):
										for k in range(layerInfo[3]):
											convSum += K[n0][i][j][k] * A[i+x0][j+y0][k]
								pixel.append(max(0, convSum))
							row.append(pixel)
						Aprime.append(row)

					A = Aprime

				else:
					A = MaxPool(A)
					ls //= vector([2, 2])

			#percentComplete += 50/256
			#print("CNN  complete - " + str(percentComplete))

			# running the LSTM cell
			hx = vector(h.data + [entry for mrx in A for vec in mrx for entry in vec])

			f = (self.Wf * hx + self.bf).sigmoid()
			i = (self.Wi * hx + self.bi).sigmoid()
			tC = (self.Wc * hx + self.bc).tanh()
			o = (self.Wo * hx + self.bo).sigmoid()

			C = (f * C) + (i * tC)
			h = o * C.tanh()

			#percentComplete += 50/256
			#print("LSTM complete - " + str(percentComplete))

		a = C.sigmoid()
		return maxI(a.data)

	def advData(self, Image):
		# LSTM run values
		C_data = list()
		hx_data = list()

		f_data = list([])
		i_data = list([])
		c_data = list([])
		o_data = list([])

		zf_data = list([])
		zi_data = list([])
		zc_data = list([])
		zo_data = list([])

		K_data = self.kVals # K_data[layer][kernelNum] = rank 3 raw tensor
		A_data = list() # A_data[cell][layer] = rank 3 raw tensor
		Z_data = list() # Z_data[cell][layer] = rank 3 raw tensor

		C = vector([0]*self.n)
		h = vector([0]*self.n)
		C_data.append(C)
		for cellNum in range(len(Image)):
			# running the CNN
			A_cell = list()
			Z_cell = list()
			A = Image[cellNum]
			ls = vector([len(Image[cellNum]), len(Image[cellNum][0])]) # x, y
			for gamma in range(len(self.CNNstruct)):
				Z = list()
				A_cell.append(A)
				if self.order[gamma]:
					K = self.kVals[gamma]
					layerInfo = self.CNNstruct[gamma]
					ls -= vector([layerInfo[1], layerInfo[2]]) - vector([1, 1])

					Aprime = list()
					for x0 in range(ls.data[0]):
						row = list()
						zrow = list()
						for y0 in range(ls.data[1]):
							pixel = list()
							zpixel = list()
							for n0 in range(layerInfo[0]):
								convSum = 0
								for i in range(layerInfo[1]):
									for j in range(layerInfo[2]):
										for k in range(layerInfo[3]):
											convSum += K[n0][i][j][k] * A[i+x0][j+y0][k]
								pixel.append(max(0, convSum))
								zpixel.append(convSum)
							row.append(pixel)
							zrow.append(zpixel)
						Aprime.append(row)
						Z.append(zrow)
					A = Aprime

				else:
					A = MaxPool(A)
					ls //= vector([2, 2])
					Z = None
				Z_cell.append(Z)

			A_cell.append(A)

			A_data.append(A_cell[::-1])
			Z_data.append(Z_cell[::-1])

			# running the LSTM cell
			hx = vector(h.data + [entry for mrx in A for vec in mrx for entry in vec])

			C_data.append(C)
			hx_data.append(hx)

			zf = self.Wf * hx + self.bf
			zf_data.append(zf)
			zi = self.Wi * hx + self.bi
			zi_data.append(zi)
			zc = self.Wc * hx + self.bc
			zc_data.append(zc)
			zo = self.Wo * hx + self.bo
			zo_data.append(zo)

			f = zf.sigmoid()
			f_data.append(f)
			i = zi.sigmoid()
			i_data.append(i)
			tC = zc.tanh()
			c_data.append(tC)
			o = zo.sigmoid()
			o_data.append(o)

			C = (f * C) + (i * tC)
			h = o * C.tanh()
		hx_data.append(vector([0]*self.m))
		a = C.sigmoid()
		return [
			C_data,
			hx_data,
			f_data,
			i_data,
			c_data,
			o_data,
			zf_data,
			zi_data,
			zc_data,
			zo_data,
			K_data,
			A_data,
			Z_data,
			a
		]

	def getError(self, a, y):
		return

	def __runConvNet__(self, image):
		convNum = 0
		poolNum = 0
		for layerType in self.__ordering__:
			if layerType:
				# maxpool layer
				image = maxpool(image, self.__poolInfo__[poolNum][0], self.__poolInfo__[poolNum][1])
				poolNum += 1
			else:
				# conv layer
				image = conv(image, self.__kVals__[convNum], self.__kStride__[convNum])
				convNum += 1
		return image

def maxI(x):
	I = 0
	for i in range(len(x)):
		if x[i] > x[I]:
			I = i
	return I