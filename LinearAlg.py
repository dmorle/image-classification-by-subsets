import math
import sys

def MrxPrint(matrix):
	for row in matrix:
		Rstr = "["
		for elm in row:
			elmLen = len(str(elm))
			if elmLen > 9:
				elmLen = 9
			Rstr += str(elm)[0:elmLen]
			for i in range(9 - elmLen):
				Rstr += " "
			Rstr += ", "
		print(Rstr + "]")
	print("\n")
	return

def floatElim(matrix, augment):
	size = len(matrix)
	for i in range(size):
		if matrix[i][i] == 0:
			for j in range(i + 1, size):
				if matrix[j][i] != 0:
					temp = (matrix[j].copy(), augment[j])
					matrix[j] = matrix[i]
					augment[j] = augment[i]
					matrix[i], augment[i] = temp
					del temp
					break
			if matrix[i][i] == 0:
				raise(Exception("Something went wrong..."))
		if matrix[i][i] != 1:
			multiple = matrix[i][i]
			for n in range(i, size):
				matrix[i][n]/= multiple
			augment[i]/=multiple
		for j in range(size): 
			if j != i:
				if matrix[j][i] != 0:
					multiple = matrix[j][i]
					for n in range(i, size):
						matrix[j][n] -= multiple * matrix[i][n]
					augment[j] -= multiple * augment[i]
	return(matrix, augment)

def MrxInv(mrx):
	size = len(mrx)
	mrxI = list()
	for i in range(size):
		row = list()
		for j in range(size):
			if i==j:
				row.append(1)
			else:
				row.append(0)
		mrxI.append(row)

	for i in range(size):
		if mrx[i][i] == 0:
			for j in range(i + 1, size):
				if mrx[j][i] != 0:
					mrx = rowSwitch(mrx, i, j)
					mrxI = rowSwitch(mrxI, i, j)
					break
			if mrx[i][i] == 0:
				raise(Exception("Inverse matrix does not exist"))
		if mrx[i][i] != 1:
			scalar = 1/mrx[i][i]
			mrx = rowScale(mrx, i, scalar)
			mrxI = rowScale(mrxI, i, scalar)
		for j in range(size): 
			if j != i:
				if mrx[j][i] != 0:
					scale = -mrx[j][i]
					mrx = rowSum(mrx, i, j, scale)
					mrxI = rowSum(mrxI, i, j, scale)
	return mrxI

def rowScale(mrx, rowNum, scale):
	ret = mrx.copy()
	row = mrx[rowNum]
	ret[rowNum] = VctScalar(row, scale)
	return ret

def rowSum(mrx, aRowNum, bRowNum, scale):
	ret = mrx.copy()
	row = mrx[aRowNum].copy()
	row = VctScalar(row, scale)
	ret[bRowNum] = VctSum(mrx[bRowNum], row)
	return ret

def rowSwitch(mrx, r1, r2):
	temp = mrx[r1].copy()
	mrx[r1] = mrx[r2]
	mrx[r2] = temp
	return mrx

def MrxProd(m1, m2):
	ret = []
	# m is len(m1), n is len(m1[0]) = len(m2), s is len(m2[0])
	# ret is mxs
	for i in m1:
		row = []
		for j in range(len(m2[0])):
			entry = 0
			for l in range(len(i)):
				entry += i[l]*m2[l][j]
			row += [entry]
		ret.append(row)
	return ret

def MrxTrans(mrx):
	ret = []
	for i in range(len(mrx[0])):
		row = []
		for j in range(len(mrx)):
			row += [mrx[j][i]]
		ret.append(row)
	return ret

def MrxScalar(mrx, c):
	ret = mrx.copy()
	for i in range(len(ret)):
		ret[i] = VctScalar(ret[i], c)
	return ret

def MrxSum(*args):
	ret = []
	for i in range(len(args[0])):
		ret.append([])
		for j in range(len(args[0][0])):
			ret[i].append(0)
			for mrx in args:
				ret[i][j] += mrx[i][j]
	return ret

def MrxSumLst(lst):
	ret = []
	for i in range(len(lst[0])):
		ret.append([])
		for j in range(len(lst[0][0])):
			ret[i].append(0)
			for mrx in lst:
				ret[i][j] += mrx[i][j]
	return ret

def getMrx(vct):
	return [vct]

def VctTrans(vct):
	ret = list()
	for i in vct:
		ret.append([i])
	return ret

def PWProd(vct1, vct2):
	ret = list()
	for i in range(len(vct1)):
		ret.append(vct1[i]*vct2[i])
	return ret

def VctScalar(vct, c):
	ret = list(vct).copy()
	for i in range(len(vct)):
		ret[i] *= c
	return ret

def VctLength(vct):
	ret = 0
	for i in vct:
		ret += i*i
	return ret**(1/2)

def VctSum(*args):
	ret = []
	for i in range(len(args[0])):
		ret += [0]
		for vct in args:
			ret[i] += vct[i]
	return tuple(ret)

def VctSumLst(lst):
	ret = []
	for i in range(len(lst[0])):
		ret += [0]
		for vct in lst:
			ret[i] += vct[i]
	return tuple(ret)

def VctDiff(v1, v2):
	ret = []
	for i in range(len(v1)):
		ret += [v1[i] - v2[i]]
	return tuple(ret)

def DotProd(v1, v2):
	ret = 0
	for i in range(len(v1)):
		ret += v1[i]*v2[i]
	return ret

def CrossProd(v1, v2):
	return (
		v1[1]*v2[2] - v1[2]*v2[1],
		v1[2]*v2[0] - v1[0]*v2[2],
		v1[0]*v2[1] - v1[1]*v2[0]
		)

def MrxVctProd(mrx, vct):
	ret = []
	for i in mrx:
		ret += [DotProd(i, vct)]
	return tuple(ret)

def CosTheta(v1, v2):
	return

#------------------------------------ Neural Net Shit ------------------------------------#

# def get_linenumber():
#     cf = currentframe()
#     return cf.f_back.f_lineno

def sigmoid(x):
	if type(x) is int or type(x) is float:
		exp = 0
		try:exp = math.exp(-x)
		except OverflowError:return(0)
		return(1/(1 + exp))
	elif type(x) is list or type(x) is tuple:
		return [sigmoid(i) for i in x]

def sigmoid1(x):
	if type(x) is int or type(x) is float:
		exp = 0
		try:exp = math.exp(-x)
		except OverflowError:return(0)
		try: exp2=(1+exp)**2
		except OverflowError:return(0)
		return(exp/exp2)
	elif type(x) is list or type(x) is tuple:
		return [sigmoid1(i) for i in x]