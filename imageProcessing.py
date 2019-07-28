import os
import pickle
from PIL import Image
from hilbertCurve import hilbertCurve

def deleteFolder(path):
	if os.path.isdir(path):
		for file in os.listdir(path):
			if (os.path.isfile(path + file)):
				os.remove(path + file)
			else:
				deleteFolder(path + file + '\\')
		os.rmdir(path)
	return

def createDataTree(initPath, newPath = ""):
	newPath = newPath + "trainingData\\"
	os.mkdir(newPath)
	for category in os.listdir(initPath):
		if category[3] == '.':
			os.mkdir(newPath + category + '\\')

def createData():
	hC = hilbertCurve(3)
	curve = hC.getDPoints()
	for point in curve:
		point[0] = int(224 * point[0])
		point[1] = int(224 * point[1])

	initPath = "C:\\Users\\dmorl\\Desktop\\File_Folder\\Computer Vision Documents\\Data Sets\\Caltech 256\\256_ObjectCategories\\"
	newPath = "trainingData\\"

	deleteFolder(newPath)
	createDataTree(initPath)

	print("createData - Setup complete")

	for category in os.listdir(initPath):
		if category[3] == '.':
			lPath = initPath + category + '\\'
			newlPath = newPath + category + '\\'

			for fileName in os.listdir(lPath):
				if ".jpg" in fileName:
					pictureData = list()

					im = Image.open(lPath + fileName, 'r')
					size = max(im.width, im.height)
					im = im.crop((0, 0, size, size))
					im = im = im.resize((256, 256))

					for point in curve:
						subIm = im.crop((point[0], point[1], point[0] + 32, point[1] + 32))
						pixels = list(subIm.getdata())
						pictureData.append(pixels)

					with open(newlPath + fileName.split(".jpg")[0] + ".pkl", "wb") as f:
						pickle.dump(pictureData, f)
		print("createData - " + category + " complete")
	return

def formatData():
	path = "trainingData\\"
	for category in os.listdir(path):
		lPath = path + category + '\\'

		for fileName in os.listdir(lPath):
			if ".pkl" in fileName:
				oData = None
				with open(lPath + fileName, "rb") as f:
					oData = pickle.load(f)
				nData = list()
				for oldCell in oData:
					cell = list()
					for y in range(32):
						row = list()
						for x in range(32):
							row.append(oldCell[32*y + x])
						cell.append(row)
					nData.append(cell)
				with open(lPath + fileName, "wb") as f:
					pickle.dump(nData, f)
		print("formatData - " + category + " complete")
	return

def compressData():
	path = "trainingData\\"
	for category in os.listdir(path):
		lPath = path + category + '\\'

		for fileName in os.listdir(lPath):
			if ".pkl" in fileName:
				oData = None
				with open(lPath + fileName, "rb") as f:
					oData = pickle.load(f)
				nData = list()
				for oldCell in oData:
					cell = list()
					for y in range(25):
						row = list()
						for x in range(25):
							row.append(vecAverage(oldCell[2*y][2*x], oldCell[2*y+1][2*x], oldCell[2*y][2*x+1], oldCell[2*y+1][2*x+1]))
						cell.append(row)
					nData.append(cell)
				with open(lPath + fileName, "wb") as f:
					pickle.dump(nData, f)
				print(fileName + " complete")
	return

def vecAverage(*args):
	dim = len(args[0])
	ret = [0]*dim
	for vec in args:
		for i in range(dim):
			ret[i] += vec[i]
	for i in range(dim):
		ret[i] /= len(args)
	return ret

def average(*args):
	ret = 0
	for val in args:
		ret += val
	return ret / len(args)

def main():
	print("Creating Data")
	createData()
	print("createData complete")
	print("Formatting Data")
	formatData()
	print("formatData complete")
	return

if __name__ == "__main__":
	main()