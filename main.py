import pickle
import os
from neuralNetwork import *
from GDAlgorithm import *

def createNetwork():
	CNNstruct = [		#32
		[10, 5, 5, 3],	#28
		[],				#14
		[25, 3, 3, 10],	#12
		[],				#6
		[50, 3, 3, 25],	#4
		[],				#2
		[100, 2, 2, 50]	#1
	]
	return NeuralNetwork(CNNstruct, n = 10, p = 100)

def loadNetwork():
	with open("NeuralNet2.pkl", "rb") as f:
		return pickle.load(f)

def saveNetwork(NN):
	with open("NeuralNet2.pkl", "wb") as f:
		pickle.dump(NN, f)

def loadBatch(numberOfLabels, imagesPerLabel, batchNum):
	data_batch = list()
	y_batch = list()

	labels = os.listdir("trainingData\\")

	GenY = [0]*257
	for lNum in range(numberOfLabels):

		imageNames = os.listdir("trainingData\\" + labels[lNum] + '\\')
		y = GenY.copy()
		y[lNum] = 1

		for iNum in range(batchNum, imagesPerLabel + batchNum):
			with open("trainingData\\" + labels[lNum] + '\\' + imageNames[iNum], "rb") as f:
				data_batch.append(pickle.load(f))
			y_batch.append(y)

	return (data_batch, y_batch)

def main():
	NN = createNetwork()
	print("network created")
	# NN = loadNetwork()
	# print("network loaded")

	for batchNum in range(0, 4):
		(data_batch, y_batch) = loadBatch(10, 1, batchNum)
		print("loaded training data")

		print("running gradient decent on batch " + str(batchNum))
		GD(NN, data_batch, y_batch, 0.01)
		print("gradient decent complete")

		saveNetwork(NN)
		print("network stored on hard drive")

	(data_batch, y_batch) = loadBatch(10, 1, 0)
	print("loaded testing data")

	for i in range(0, 10):
		print(NN.run(data_batch[i]))

	# """saveNetwork(NN)
	# print("network stored on hard drive")"""
	return

if __name__ == "__main__":
	main()