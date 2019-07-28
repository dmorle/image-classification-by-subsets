from NNLinearAlg import *

def Gradient(NN, Image, y, n, p, m):

	#--------------------------------- getting data from the NN ------------------------------------#

	[
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
	] = NN.advData(Image)

	print("Data Recieved")

	order = NN.order
	CNNstruct = NN.CNNstruct
	K_data = K_data[::-1]

	Wf = NN.Wf
	Wi = NN.Wi
	Wc = NN.Wc
	Wo = NN.Wo

	bf = NN.bf
	bi = NN.bi
	bc = NN.bc
	bo = NN.bo

	cellNum = len(Image)

	#------------------------------------ Finding LSTM Gradient ------------------------------------#

	vec = list()
	for i in range(n):
		if y[i] == 0:
			if 1-a.data[i] == 0:
				vec.append(-2.0**128)
			else:
				vec.append(-1/1-a.data[i])
		else:
			if a.data[i] == 0:
				vec.append(2.0**128)
			else:
				vec.append(y[i]/a.data[i])
	
	D_CGAMMA = vector(vec)/-n # current gradient wrt the cell state
	D_hGAMMA = vector([0]*n) # current gradient wrt the hidden state

	GradC = [None]*(cellNum+1) # list of gradients wrt the cell state over all cells
	GradH = [None]*(cellNum+1) # list of gradients wrt the hidden state over all cells
	GradX = [None]*cellNum # lost my train of thought.  Come back here later plz

	GradC[cellNum] = D_CGAMMA
	GradH[cellNum] = D_hGAMMA

	for gamma in range(cellNum-1, -1, -1):
		nD_CGAMMA = vector([0]*n) # Gradient wrt the cell state at gamma-1 ; continues to be computed through the loop
		nD_hxGAMMA = vector([0]*m) # Gradient wrt the hidden state at gamma-1 ; continues to be computed through the loop
		for i in range(n):
			Wfi = list()
			Wii = list()
			Wci = list()
			Woi = list()
			for j in range(m):
				Wfi.append(Wf.data[i][j])
				Wii.append(Wi.data[i][j])
				Wci.append(Wc.data[i][j])
				Woi.append(Wi.data[i][j])
			Wfi = vector(Wfi)
			Wii = vector(Wii)
			Wci = vector(Wci)
			Woi = vector(Woi)

			# gradient of the cell state at i wrt the hx at gamma-1
			grad_Ci_hx = (
				Wfi * C_data[gamma].data[i] * sigmoid1(zf_data[gamma].data[i]) + 
				Wii * c_data[gamma].data[i] * sigmoid1(zi_data[gamma].data[i]) +
				Wci * i_data[gamma].data[i] * tanh1(zc_data[gamma].data[i])
			)
			# gradient of the hidden state at i wrt the hx at gamma-1
			grad_hi_hx = (
				grad_Ci_hx * o_data[gamma].data[i] * tanh1(C_data[gamma+1].data[i]) +
				Woi * tanh(C_data[gamma+1].data[i]) * sigmoid1(zo_data[gamma].data[i])
			)

			nD_hxGAMMA += grad_Ci_hx * D_CGAMMA.data[i] + grad_hi_hx * D_hGAMMA.data[i]

		nD_CGAMMA = f_data[gamma] * (D_CGAMMA + D_hGAMMA * o_data[gamma] * C_data[gamma+1].tanh1())

		# transitioning between cells to continue backpropogation
		D_hGAMMA_vec = list()
		D_xGAMMA_vec = list()
		for i in range(m):
			if i < n:
				D_hGAMMA_vec.append(nD_hxGAMMA.data[i])
			else:
				D_xGAMMA_vec.append(nD_hxGAMMA.data[i])

		D_CGAMMA = nD_CGAMMA
		D_hGAMMA = vector(D_hGAMMA_vec)

		# Note, this section was indexed with C & H @ [gamma-1] and X @ gamma
		GradC[gamma] = D_CGAMMA
		GradH[gamma] = D_hGAMMA
		GradX[gamma] = vector(D_xGAMMA_vec)

	#----------------------------------- cell gradients complete -----------------------------------#

	#---------------------------------- Weight and bias gradients ----------------------------------#

	grad_bf = vector([0]*n)
	grad_bi = vector([0]*n)
	grad_bc = vector([0]*n)
	grad_bo = vector([0]*n)
	grad_Wf = matrix([[0]*m]*n)
	grad_Wi = matrix([[0]*m]*n)
	grad_Wc = matrix([[0]*m]*n)
	grad_Wo = matrix([[0]*m]*n)
	for gamma in range(0, cellNum):
		commonVec = GradC[gamma+1] + GradH[gamma+1] * o_data[gamma] * C_data[gamma+1].tanh1()

									# gradient wrt the biases #

		D_bfGAMMA = C_data[gamma] * zf_data[gamma].sigmoid1() * commonVec
		D_biGAMMA = c_data[gamma] * zi_data[gamma].sigmoid1() * commonVec
		D_bcGAMMA = i_data[gamma] * zc_data[gamma].tanh1() * commonVec
		D_boGAMMA = GradH[gamma+1] * C_data[gamma+1].tanh() * zo_data[gamma].sigmoid1()

		grad_bf += D_bfGAMMA
		grad_bi += D_biGAMMA
		grad_bc += D_bcGAMMA
		grad_bo += D_boGAMMA

									# gradient wrt the weights #
		
		hxVec = hx_data[gamma]
		for a in range(n):
			for b in range(m):
				grad_Wf.data[a][b] += D_bfGAMMA.data[a] * hxVec.data[b]
				grad_Wi.data[a][b] += D_biGAMMA.data[a] * hxVec.data[b]
				grad_Wc.data[a][b] += D_bcGAMMA.data[a] * hxVec.data[b]
				grad_Wo.data[a][b] += D_boGAMMA.data[a] * hxVec.data[b]

	#------------------------------------ LSTM Gradient Complete -----------------------------------#

	#------------------------------------- Finding CNN Gradient ------------------------------------#

	ls_data = list()
	ls = vector([len(Image[0]), len(Image[0][0])]) # x, y
	for l in range(len(order)):
		ls_data.append(ls.data)
		if order[l]:
			ls -= vector([CNNstruct[l][1], CNNstruct[l][2]]) - vector([1, 1])
		else:
			ls //= vector([2, 2])

	ls_data.append(ls.data)
	ls_data = ls_data[::-1]
	order = order[::-1]
	CNNstruct = CNNstruct[::-1]

	GradK = list()
	for cell in range(len(GradX)):
		D_xGAMMA = GradX[cell].data # Output from the LSTM gradient

		D_AGAMMA = list() # Activation gradient over all layers for a particular gamma
		D_KGAMMA = list() # Kernel gradient over all layers and kernels at gamma

		D_AlGAMMA = [[D_xGAMMA]] # current rank 3 tensor gradient for activations at gamma at l + 1
		D_AGAMMA.append(D_AlGAMMA)

		for l in range(len(order)):
			layerInfo = CNNstruct[l]

			nD_AlGAMMA = list() # rank 3 tensor gradient for activations at gamma at l ; to be computed using nabla(l+1)
			D_KlGAMMA = list() # kernel gradient for over all kernels at gamma at l

			if order[l]: # missing optimization for MAXPOOL gradient
				# activation gradient
				for x in range(ls_data[l+1][0]):
					row = list()
					for y in range(ls_data[l+1][1]):
						pixel = list()
						for z in range(layerInfo[3]):
							channel = 0
							#construction of nD_AlGAMMA
							for i in range(ls_data[l][0]):
								for j in range(ls_data[l][1]):
									for k in range(layerInfo[0]):
										if 0 <= x-i and x-i < layerInfo[1] and 0 <= y-j and y-j < layerInfo[2]:
											channel += D_AlGAMMA[i][j][k] * ReLU1(Z_data[cell][l][i][j][k]) * K_data[l][k][x-i][y-j][z]
							pixel.append(channel)
						row.append(pixel)
					nD_AlGAMMA.append(row)

				# kernel gradient
				for n in range(layerInfo[0]):
					kernel = list()
					for x in range(layerInfo[1]):
						row = list()
						for y in range(layerInfo[2]):
							pixel = list()
							for z in range(layerInfo[3]):
								channel = 0
								#construction of D_KlGAMMA
								for i in range(ls_data[l][0]):
									for j in range(ls_data[l][1]):
										# Note A @ l+1 = ReLU( Z @ l ).  NOT L+1.  FUUUUCCCKKKing kmn
										channel += D_AlGAMMA[i][j][n] * ReLU1(Z_data[cell][l][i][j][n]) * A_data[cell][l+1][x+i][y+j][z]
								pixel.append(channel)
							row.append(pixel)
						kernel.append(row)
					D_KlGAMMA.append(kernel)

			else:
				# MAXPOOL backpropogation
				# iterate over A[l+1] and check if maxpool gives the coorosponding A[l]
				for x in range(ls_data[l+1][0]):
					D_row = list()
					for y in range(ls_data[l+1][1]):
						D_pixel = list()
						for z in range(CNNstruct[l+1][0]):
							if A_data[cell][l+1][x][y][z] == A_data[cell][l][x//2][y//2][z]:
								D_pixel.append(D_AlGAMMA[x//2][y//2][z])
							else:
								D_pixel.append(0)
						D_row.append(D_pixel)
					nD_AlGAMMA.append(D_row)
				D_KlGAMMA = None

			D_AlGAMMA = nD_AlGAMMA
			D_AGAMMA.append(D_AlGAMMA)
			D_KGAMMA.append(D_KlGAMMA)
		GradK.append(D_KGAMMA[::-1])

	#------------------------------------ CNN Gradient Complete ------------------------------------#

	return (
			GradK,
			grad_bf,
			grad_bi,
			grad_bc,
			grad_bo,
			grad_Wf,
			grad_Wi,
			grad_Wc,
			grad_Wo
		)

def GD(NN, data_batch, y_batch, alpha):

	n = NN.n
	p = NN.p
	m = NN.m

	BatchGradK = list()
	grad_bf = vector([0]*n)
	grad_bi = vector([0]*n)
	grad_bc = vector([0]*n)
	grad_bo = vector([0]*n)
	grad_Wf = matrix([[0]*m]*n)
	grad_Wi = matrix([[0]*m]*n)
	grad_Wc = matrix([[0]*m]*n)
	grad_Wo = matrix([[0]*m]*n)
	for num in range(len(data_batch)):
		print("GD - Image " + str(num) + " started")

		(
			GradK_n,
			grad_bf_n,
			grad_bi_n,
			grad_bc_n,
			grad_bo_n,
			grad_Wf_n,
			grad_Wi_n,
			grad_Wc_n,
			grad_Wo_n
		) = Gradient(NN, data_batch[num], y_batch[num], n, p, m)
		BatchGradK.append(GradK_n)
		grad_bf = grad_bf + grad_bf_n
		grad_bi = grad_bi + grad_bi_n
		grad_bc = grad_bc + grad_bc_n
		grad_bo = grad_bo + grad_bo_n
		grad_Wf = grad_Wf + grad_Wf_n
		grad_Wi = grad_Wi + grad_Wi_n
		grad_Wc = grad_Wc + grad_Wc_n
		grad_Wo = grad_Wo + grad_Wo_n

		print("GD - Image " + str(num) + " complete")
	print("GD - Updating Network")

	# Updating NN.kVals

	for GradK in BatchGradK:
		for cell in GradK:
			for l in range(len(cell)):
				if cell[l]:
					for k in range(len(cell[l])):
						for r in range(len(cell[l][k])):
							for p in range(len(cell[l][k][r])):
								for c in range(len(cell[l][k][r][p])):
									NN.kVals[l][k][r][p][c] -= alpha * cell[l][k][r][p][c]

	# Updating LSTM parameters

	NN.Wf = NN.Wf - (grad_Wf * alpha)
	NN.Wi = NN.Wi - (grad_Wi * alpha)
	NN.Wc = NN.Wc - (grad_Wc * alpha)
	NN.Wo = NN.Wo - (grad_Wo * alpha)
	NN.bf = NN.bf - (grad_bf * alpha)
	NN.bi = NN.bi - (grad_bi * alpha)
	NN.bc = NN.bc - (grad_bc * alpha)
	NN.bo = NN.bo - (grad_bo * alpha)

	print("GD - Update complete")

	#---------------------------------- Gradient Decent Complete -----------------------------------#

	return

def sigmoid(val):
	en = 0
	try:
		en = math.exp(-val)
	except OverflowError:
		return 0
	else:
		return 1/(1 + en)

def sigmoid1(val):
	ep = 0
	try:
		ep = math.exp(val)
	except OverflowError:
		return 0
	else:
		en = 0
		try:
			math.exp(-val)
		except OverflowError:
			return 0
		else:
			return 1/(ep + 2 + en)

def tanh(val):
	ep = 0
	try:
		ep = math.exp(val)
	except OverflowError:
		return 1
	else:
		en = 0
		try:
			en = math.exp(-val)
		except OverflowError:
			return -1
		else:
			return (ep - en)/(ep + en)

def tanh1(val):
	val = tanh(val)
	return 1 - val*val

def ReLU(x):
	return max(0, x)

def ReLU1(x):
	if x <= 0:
		return 0
	return 1