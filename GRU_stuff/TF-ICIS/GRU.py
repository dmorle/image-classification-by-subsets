import tensorflow as tf
import numpy as np

def sig1(x):
	x = tf.nn.sigmoid(x)
	y = tf.subtract(1, x)
	return tf.multiply(x, y)

def tanh1(x):
	x = tf.nn.tanh(x)
	x = tf.multiply(x, x)
	return tf.subtract(1, x)

def matHadamard(T, v):
	shape = T.shape
	m = shape[0]
	n = shape[1]
	T = tf.tensordot(v.reshape(m), T, 0)
	U_Tensor = np.concatenate([np.identity(m).reshape(m, m, 1) for i in range(n)], 2)
	T = tf.multiply(T, U_Tensor)
	T = tf.reduce_sum(T, 0)
	return T

# returns
# (
# 	h_grad(gamma), x_grad(gamma),
# 	Wz_grad, Wr_grad, Wp_grad,
# 	Uz_grad, Ur_grad, Up_grad,
# 	bz_grad, br_grad, bp_grad
# )
def backPropGRU_Cell(
	Wz, Wr, Wp, 
	Uz, Ur, Up, 
	bz, br, bp,
	Z, ZT, R, RT, P, PT,
	H, X,
	h_grad, l = None
	name = None
):
	if name is None and l is not None:
		name = "backprop_GRU_Cell_" + str(l)
	elif name is None:
		name = "backprop_GRU_Cell"

	with tf.name_scope("Gradient_Setup")
		xt = tf.transpose(X)
		ht = tf.transpose(H)
		rht = tf.transpose(tf.multiply(R, H))
		Upt = tf.transpose(Up)

		hsig_r = tf.multiply(H, sig1(RT))

		zMult = tf.multiply(
			tf.multiply(
				h_grad,
				tf.sub(P, H)
			),
			sig1(ZT)
		)
		pMult = tf.multiply(
			tf.multiply(
				h_grad,
				Z
			),
			tanh1(PT)
		)
		rMult = tf.multiply(
			tf.matmul(
				Upt,
				u2
			),
			hsig_r
		)

	with tf.name_scope("Z_Var_Grad"):
		bz_grad = tf.identity(zMult)
		Wz_grad = tf.matmul(zMult, xt)
		Uz_grad = tf.matmul(zMult, ht)

	with tf.name_scope("R_Var_Grad"):
		br_grad = tf.identity(rMult)
		Wr_grad = tf.matmul(rMult, xt)
		Ur_grad = tf.matmul(rMult, ht)

	with tf.name_scope("P_Var_Grad"):
		bp_grad = tf.identity(pMult)
		Wp_grad = tf.matmul(pMult, xt)
		Up_grad = tf.matmul(pMult, rht)

	with tf.name_scope("x_Grad"):
		x_grad = tf.add(
			tf.add(
				tf.matmul(
					tf.transpose(Wz),
					zMult
				),
				tf.matmul(
					tf.transpose(Wp),
					pMult
				)
			),
			tf.matmul(
				tf.transpose(
					tf.matmul(
						Up,
						matHadamard(
							Wr,
							hsig_r
						)
					)
				),
				pMult
			)
		)

	with tf.name_scope("h_Grad"):
		h_grad = tf.add(
			tf.add(
				tf.add(
					tf.multiply(
						h_grad, 
						tf.sub(
							1,
							Z
						)
					),
					tf.matmul(
						tf.transpose(Uz),
						zMult
					)
				),
				tf.matmul(
					Upt, 
					pMult
				)
			),
			tf.matmul(
				tf.transpose(
					tf.matmul(
						Up, 
						matHadamard(
							Ur, 
							hsig_r
						)
					)
				), 
				pMult
			)	
		)

	return

# returns h(gamma + 1)
def GRU_Cell(
	M, N, h, x, 
	Wz, Wr, Wp, 
	Uz, Ur, Up, 
	bz, br, bp, 
	Z_Stack, ZT_Stack, R_Stack, RT_Stack, P_Stack, PT_Stack, H_Stack,
	name = "GRU_Cell"
):
	with tf.name_scope("Z_FC_Layer"):
		z_tilda = tf.add(tf.add(
			tf.matmul(Wz, x, name = "zWx"), 
			tf.matmul(Uz, h, name = "zUh")),
			bz
		)
		z = tf.nn.sigmoid(z_tilda)
		ZT_Stack.append(z_tilda)
		Z_Stack.append(z)

	with tf.name_scope("R_FC_Layer"):
		r_tilda = tf.add(tf.add(
			tf.matmul(Wr, x, name = "rWx"), 
			tf.matmul(Ur, h, name = "rUh")),
			br
		)
		r = tf.nn.sigmoid(r_tilda)
		RT_Stack.append(r_tilda)
		R_Stack.append(r)

	with tf.name_scope("P_FC_Layer"):
		p_tilda = tf.add(tf.add(
			tf.matmul(Wp, x, name = "pWx"), 
			tf.matmul(
				Up, 
				tf.multiply(r, h, name = "rh"), 
				name = "pUh")),
			bp
		)
		p = tf.nn.tanh(p_tilda)
		PT_Stack.append(p_tilda)
		P_Stack.append(p)

	oneVec = tf.ones(name = "1Vec", shape = (N, 1), dtype = tf.float32)
	
	with tf.name_scope(name):
		h = tf.add(
			tf.multiply(tf.math.subtract(oneVec, z, name = "z_Inv"), h, name = "zh"),
			tf.multiply(z, p, name = "zp")
		)
		H_Stack.append(h)
		return h
