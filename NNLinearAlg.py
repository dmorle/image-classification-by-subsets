import math

def DotProd(v1, v2):
	ret = 0
	for i in range(len(v1)):
		ret += v1[i]*v2[i]
	return ret

class vector:
	def __init__(self, data):
		self.data = list(data)
		return

	def __str__(self):
		return str(self.data)

	def __eq__(self, vec):
		if type(vec) is not vector:
			return False
		if self.size() != vec.size():
			return False
		for i in range(vec.size()):
			if self.data[i] != vec.data[i]:
				return False
		return True

	def __ne__(self, vec):
		if type(vec) is not vector:
			return True
		if self.size() != vec.size():
			return True
		for i in range(vec.size()):
			if self.data[i] != vec.data[i]:
				return True
		return False

	def __add__(self, other):
		if type(other) is int or type(other) is float:
			return vector(self.data[i]+other for i in range(self.size()))
		if type(other) is vector:
			return vector(self.data[i]+other.data[i] for i in range(self.size()))
		raise(Exception("Invalid Argument"))
		return

	def __iadd__(self, other):
		if type(other) is int or type(other) is float:
			return vector(self.data[i]+other for i in range(self.size()))
		if type(other) is vector:
			return vector(self.data[i]+other.data[i] for i in range(self.size()))
		raise(Exception("Invalid Argument"))
		return

	def __sub__(self, other):
		if type(other) is int or type(other) is float:
			return vector(self.data[i]-other for i in range(self.size()))
		if type(other) is vector:
			return vector(self.data[i]-other.data[i] for i in range(self.size()))
		raise(Exception("Invalid Argument"))
		return

	def __isub__(self, other):
		if type(other) is int or type(other) is float:
			return vector(self.data[i]-other for i in range(self.size()))
		if type(other) is vector:
			return vector(self.data[i]-other.data[i] for i in range(self.size()))
		raise(Exception("Invalid Argument"))
		return

	# pointwise operation
	def __mul__(self, other):
		if type(other) is int or type(other) is float:
			return vector(other * self.data[i] for i in range(self.size()))
		if type(other) is vector:
			return vector(other.data[i] * self.data[i] for i in range(self.size()))
		if type(other) is matrix:
			mrx = list()
			for i in range(self.size()):
				row = list()
				for j in range(other.size().data[1]):
					row.append(self.data[i] * other.data[i][j])
				mrx.append(row)
			return matrix(mrx)
		raise(Exception("Invalid Argument"))
		return

	def __truediv__(self, other):
		if type(other) is int or type(other) is float:
			return vector(self.data[i] / other for i in range(self.size()))
		if type(other) is vector:
			return vector(self.data[i] / other.data[i] for i in range(self.size()))
		raise(Exception("Invalid Argument"))
		return

	def __itruediv__(self, other):
		if type(other) is int or type(other) is float:
			return vector(self.data[i] / other for i in range(self.size()))
		if type(other) is vector:
			return vector(self.data[i] / other.data[i] for i in range(self.size()))
		raise(Exception("Invalid Argument"))
		return

	def __floordiv__(self, other):
		if type(other) is int or type(other) is float:
			return vector(int(self.data[i] / other) for i in range(self.size()))
		if type(other) is vector:
			return vector(int(self.data[i] / other.data[i]) for i in range(self.size()))
		raise(Exception("Invalid Argument"))
		return

	def __ifloordiv__(self, other):
		if type(other) is int or type(other) is float:
			return vector(int(self.data[i] / other) for i in range(self.size()))
		if type(other) is vector:
			return vector(int(self.data[i] / other.data[i]) for i in range(self.size()))
		raise(Exception("Invalid Argument"))
		return

	def size(self):
		return len(self.data)

	def length(self):
		ret = 0
		for val in self.data:
			ret += val**2
		return ret

	def sigmoid(self):
		vec = list()
		for val in self.data:
			en = 0
			try:
				en = math.exp(-val)
			except OverflowError:
				vec.append(0)
			else:
				vec.append(1/(1 + en))
		return vector(vec)

	def sigmoid1(self):
		vec = list()
		for val in self.data:
			ep = 0
			try:
				ep = math.exp(val)
			except OverflowError:
				vec.append(0)
			else:
				en = 0
				try:
					math.exp(-val)
				except OverflowError:
					vec.append(0)
				else:
					vec.append(1/(ep + 2 + en))
		return vector(vec)

	def tanh(self):
		vec = list()
		for val in self.data:
			ep = 0
			try:
				ep = math.exp(val)
			except OverflowError:
				vec.append(1)
			else:
				en = 0
				try:
					en = math.exp(-val)
				except OverflowError:
					vec.append(-1)
				else:
					vec.append((ep - en)/(ep + en))
		return vector(vec)

	def tanh1(self):
		vec = list()
		for val in self.data:
			ep = 0
			try:
				ep = math.exp(val)
			except OverflowError:
				vec.append(0)
			else:
				en = 0
				try:
					math.exp(-val)
				except OverflowError:
					vec.append(0)
				else:
					val = (ep - en)/(ep + en)
					vec.append(1-val*val)
		return vector(vec)

	def dotProd(self, vec):
		val = 0
		for i in range(len(self.data)):
			val += self.data[i] * vec.data[i]
		return val

	def crossProd(self, vec):
		return

class matrix:
	def __init__(self, data):
		self.data = list(data)
		for i in range(self.size().data[0]):
			self.data[i] = list(self.data[i])
		self.data = list(data)
		return

	def __eq__(self, mrx):
		if type(mrx) is matrix:
			return False
		size = self.size()
		if size != mrx.size():
			return False
		for i in range(size):
			for j in range(size[0]):
				if self.data[i][j] != mrx.data[i][j]:
					return False
		return True

	def __ne__(self, mrx):
		if type(mrx) is matrix:
			return True
		size = self.size()
		if size != mrx.size():
			return True
		for i in range(size):
			for j in range(size[0]):
				if self.data[i][j] != mrx.data[i][j]:
					return True
		return False

	def __add__(self, other):
		size = self.size()
		if size == other.size():
			size = size.data
			mrx = list()
			for i in range(size[0]):
				row = list()
				for j in range(size[1]):
					row.append(self.data[i][j] + other.data[i][j])
				mrx.append(row)
			return matrix(mrx)
		raise(Exception("Incompadable Matrix"))

	def __iadd__(self, other):
		size = self.size()
		if size == other.size():
			mrx = list()
			for i in range(size[0]):
				row = list()
				for j in range(size[1]):
					row.append(self.data[i][j] + other.data[i][j])
				mrx.append(row)
			return matrix(mrx)
		raise(Exception("Incompadable Matrix"))

	def __sub__(self, other):
		size = self.size()
		if size == other.size():
			size = size.data
			mrx = list()
			for i in range(size[0]):
				row = list()
				for j in range(size[1]):
					row.append(self.data[i][j] - other.data[i][j])
				mrx.append(row)
			return matrix(mrx)
		raise(Exception("Incompadable Matrix"))

	def __isub__(self, other):
		size = self.size()
		if size == other.size():
			size = size.data
			mrx = list()
			for i in range(size[0]):
				row = list()
				for j in range(size[1]):
					row.append(self.data[i][j] - other.data[i][j])
				mrx.append(row)
			return matrix(mrx)
		raise(Exception("Incompadable Matrix"))

	def __mul__(self, other):
		if type(other) is int or type(other) is float:
			mrx = list()
			size = self.size().data
			for i in range(size[0]):
				row = list()
				for j in range(size[1]):
					row.append(self.data[i][j] * other)
				mrx.append(row)
			return matrix(mrx)
		if type(other) is vector:
			size = self.size()
			vec = list()
			for i in range(size.data[0]):
				entry = 0
				for j in range(size.data[1]):
					entry += self.data[i][j]*other.data[j]
				vec.append(entry)
			return vector(vec)
		if type(other) is matrix:
			other = other.getTranspose()
			mrx = list()
			for i in range(self.size().data[0]):
				row = list()
				for j in range(other.size().data[0]):
					row.append(DotProd(self.data[i], other.data[j]))
				mrx.append(row)
			return matrix(mrx)
		raise(Exception("Invalid Argument"))
		return

	def size(self):
		return vector([len(self.data), len(self.data[0])])

	def getTranspose(self):
		mrx = list()
		for i in range(len(self.data[0])):
			row = list()
			for j in range(len(self.data)):
				row.append(self.data[j][i])
			mrx.append(row)
		return matrix(mrx)

	def floatElim(self, vec):
		matrix = self.data
		argument = vec.data
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
					raise(Exception("Non-standard matrix solution"))
			if matrix[i][i] != 1:
				multiple = matrix[i][i]
				for n in range(i, size):
					matrix[i][n] /= multiple
				augment[i] /=multiple
			for j in range(size): 
				if j != i:
					if matrix[j][i] != 0:
						multiple = matrix[j][i]
						for n in range(i, size):
							matrix[j][n] -= multiple * matrix[i][n]
						augment[j] -= multiple * augment[i]
		return(matrix, augment)

def main():
	return

if __name__ == "__main__":
	mrx1 = matrix([
		[1, 0, 0],
		[0, 2, 0],
		[0, 0, 1]
		])

	mrx2 = matrix([
		[1, 0, 1],
		[0, 1, 1],
		[0, 0, 1]
		])

	mrx3 = mrx1*mrx2
	print(mrx3.data)
	main()