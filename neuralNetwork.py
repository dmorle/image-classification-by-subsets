import random
import pickle
from GDAlgorithm import *
from NNLinearAlg import *

"""
CNNstruct is a list which defines the structure of the convolutional neural network:
	[
		lists containing the size of kernels at the indexed layer;
			# of kernels, x, y, z
	]
An empty entry denotes a MaxPool layer

"""

class NeuralNetwork:
	def __init__(self, CNNstruct, n = 257, p = 100):
		#declaring the CNN and initializing it with random kernels
		self.CNNstruct = CNNstruct
		self.kVals = list()
		self.order = list()
		for layerInfo in CNNstruct:
			if len(layerInfo) == 4:
				self.order.append(True)
				layer = list()
				for kNum in range(layerInfo[0]):
					kernel = list()
					for x in range(layerInfo[1]):
						row = list()
						for y in range(layerInfo[2]):
							pixel = list()
							for z in range(layerInfo[3]):
								pixel.append(random.uniform(-1, 1))
							row.append(pixel)
						kernel.append(row)
					layer.append(kernel)
				self.kVals.append(layer)
			else:
				self.order.append(False)
				self.kVals.append(None)

		#declaring the LSTM and initializing it with random weights and biases
		self.n = n
		self.p = p
		self.m = self.n + self.p

		self.Wf, self.Wi, self.Wc, self.Wo = (list() for i in range(4))
		self.bf, self.bi, self.bc, self.bo = (list() for i in range(4))
		for i in range(self.n):
			self.bf.append(random.uniform(-1, 1))
			self.bi.append(random.uniform(-1, 1))
			self.bc.append(random.uniform(-1, 1))
			self.bo.append(random.uniform(-1, 1))
			Wfi, Wii, Wci, Woi = (list() for i in range(4))
			for j in range(self.m):
				Wfi.append(random.uniform(-1, 1))
				Wii.append(random.uniform(-1, 1))
				Wci.append(random.uniform(-1, 1))
				Woi.append(random.uniform(-1, 1))
			self.Wf.append(Wfi)
			self.Wi.append(Wii)
			self.Wc.append(Wci)
			self.Wo.append(Woi)
		self.Wf = matrix(self.Wf)
		self.Wi = matrix(self.Wi)
		self.Wc = matrix(self.Wc)
		self.Wo = matrix(self.Wo)
		self.bf = vector(self.bf)
		self.bi = vector(self.bi)
		self.bc = vector(self.bc)
		self.bo = vector(self.bo)
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

def MaxPool(T):
	image = list()
	for i in range(0, len(T), 2):
		row = list()
		for j in range(0, len(T[0]), 2):
			pixel = list()
			for k in range(len(T[0][0])):
				vals = [
					T[i][j][k], 
					T[i+1][j][k], 
					T[i][j+1][k],
					T[i+1][j+1][k]
					]
				pixel.append(max(vals))
			row.append(pixel)
		image.append(row)
	return image

def maxI(x):
	I = 0
	for i in range(len(x)):
		if x[i] > x[I]:
			I = i
	return I