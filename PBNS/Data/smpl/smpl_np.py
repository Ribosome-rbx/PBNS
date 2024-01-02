import sys
import torch
import numpy as np
import pickle
import smplx
import torch.nn.functional as F

class SMPLModel():
	def __init__(self, model_path, gender, rest_pose=None):
		"""
		SMPL model.

		Parameter:
		---------
		model_path: Path to the SMPL model parameters, pre-processed by
		`preprocess.py`.

		"""
		self.template = rest_pose

		self.smplx_model = smplx.create(model_path=model_path, model_type='smplx', gender=gender, num_betas=10, use_pca=True, num_pca_comps=12)
		self.use_pca = True
		
		self.J_regressor = self.smplx_model.J_regressor
		self.lbs_weights = self.smplx_model.lbs_weights
		self.v_template = self.smplx_model.v_template
		self.posedirs = self.smplx_model.posedirs
		self.left_hand_components = self.smplx_model.left_hand_components
		self.right_hand_components = self.smplx_model.right_hand_components
		self.NUM_BODY_JOINTS = self.smplx_model.NUM_BODY_JOINTS
		self.pose_mean = self.smplx_model.pose_mean
		self.shapedirs = self.smplx_model.shapedirs
		self.expr_dirs = self.smplx_model.expr_dirs
		self.parents = self.smplx_model.parents

		# load template body
		self.global_orient = torch.tensor(self.template['global_orient'], dtype=torch.float32)
		self.body_pose = torch.tensor(self.template['body_pose'], dtype=torch.float32)
		self.left_hand_pose = torch.tensor(self.template['left_hand_pose'], dtype=torch.float32)
		self.right_hand_pose = torch.tensor(self.template['right_hand_pose'], dtype=torch.float32)
		self.jaw_pose = torch.tensor(self.template['jaw_pose'], dtype=torch.float32)
		self.leye_pose = torch.tensor(self.template['leye_pose'], dtype=torch.float32)
		self.reye_pose = torch.tensor(self.template['reye_pose'], dtype=torch.float32)

		self.expression = torch.tensor(self.template['expression'], dtype=torch.float32)
		self.betas = torch.tensor(self.template['betas'], dtype=torch.float32)
		self.transl = torch.tensor(self.template['transl'], dtype=torch.float32)

		self._rest_pose = np.concatenate([self.global_orient.reshape(-1,3),self.body_pose.reshape(-1,3), np.zeros([2,3])]).reshape(-1)
		# randomly initialized hand pose

		# id_to_col = {
		# 	self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
		# }
		# self.parent = {
		# 	i: id_to_col[self.kintree_table[0, i]]
		# 	for i in range(1, self.kintree_table.shape[1])
		# }


	def forward(self, poses=None, with_body=True):
		# If no shape and pose parameters are passed along, then use the
		# ones from the module
		if not poses.shape == (72,):
			poses = torch.tensor(poses, dtype=torch.float32)
			global_orient = (poses[None,:3] if poses is not None else self.global_orient)
			body_pose = poses[None,3:66] if poses is not None else self.body_pose
			full_pose = poses.reshape(-1, 165)

			# will not be changed by poses
			betas = self.betas
			expression = self.expression
			transl = self.transl
		else:
			poses = torch.tensor(poses, dtype=torch.float32)
			global_orient = (poses[None,:3] if poses is not None else self.global_orient)
			body_pose = poses[None,3:66] if poses is not None else self.body_pose

			# will not be changed by poses
			left_hand_pose = self.left_hand_pose
			right_hand_pose = self.right_hand_pose

			jaw_pose = self.jaw_pose
			leye_pose = self.leye_pose
			reye_pose = self.reye_pose
			betas = self.betas
			expression = self.expression
			transl = self.transl

			if self.use_pca:
				left_hand_pose = torch.einsum(
					'bi,ij->bj', [left_hand_pose, self.left_hand_components])
				right_hand_pose = torch.einsum(
					'bi,ij->bj', [right_hand_pose, self.right_hand_components])

			full_pose = torch.cat([global_orient.reshape(-1, 1, 3),
									body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3),
									jaw_pose.reshape(-1, 1, 3),
									leye_pose.reshape(-1, 1, 3),
									reye_pose.reshape(-1, 1, 3),
									left_hand_pose.reshape(-1, 15, 3),
									right_hand_pose.reshape(-1, 15, 3)],
									dim=1).reshape(-1, 165)

			# Add the mean pose of the model. Does not affect the body, only the
			# hands when flat_hand_mean == False
			full_pose += self.pose_mean

		batch_size = max(betas.shape[0], global_orient.shape[0],
							body_pose.shape[0])
		# Concatenate the shape and expression coefficients
		scale = int(batch_size / betas.shape[0])
		if scale > 1:
			betas = betas.expand(scale, -1)
		shape_components = torch.cat([betas, expression], dim=-1)

		shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

		G, v =  self.lbs(shape_components, full_pose, self.v_template,
						shapedirs, self.posedirs,
						self.J_regressor, self.parents,
						self.lbs_weights, with_body=with_body,
						)
		
		if v is not None: v = v[0] + transl
		
		return G[0], v
		

	def lbs(self, betas, pose, v_template, shapedirs, posedirs,
			J_regressor, parents, lbs_weights, with_body: bool = True,):

		batch_size = max(betas.shape[0], pose.shape[0])
		device, dtype = betas.device, betas.dtype

		# Add shape contribution
		v_shaped = v_template +torch.einsum('bl,mkl->bmk', [betas, shapedirs])

		# Get the joints
		# NxJx3 array
		J = torch.einsum('bik,ji->bjk', [v_shaped, J_regressor])

		# 3. Add pose blend shapes
		# N x J x 3 x 3
		ident = torch.eye(3, dtype=dtype, device=device)

		# pose2rot == True
		rot_mats = self.batch_rodrigues(pose.view(-1, 3)).view(
			[batch_size, -1, 3, 3])

		pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
		# (N x P) x (P, V * 3) -> N x V x 3
		pose_offsets = torch.matmul(
			pose_feature, posedirs).view(batch_size, -1, 3)


		# 4. Get the global joint location
		J_transformed, A = self.batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

		v = None
		if with_body:
			v_posed = pose_offsets + v_shaped

			# 5. Do skinning:
			# W is N x V x (J + 1)
			W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
			# (N x V x (J + 1)) x (N x (J + 1) x 16)
			num_joints = J_regressor.shape[0]
			T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
				.view(batch_size, -1, 4, 4)
		
			homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
									dtype=dtype, device=device)
			v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
			v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

			v = v_homo[:, :, :3, 0]

		return A, v

	def batch_rodrigues(self, rot_vecs, epsilon = 1e-8):
		batch_size = rot_vecs.shape[0]
		device, dtype = rot_vecs.device, rot_vecs.dtype

		angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
		rot_dir = rot_vecs / angle

		cos = torch.unsqueeze(torch.cos(angle), dim=1)
		sin = torch.unsqueeze(torch.sin(angle), dim=1)

		# Bx1 arrays
		rx, ry, rz = torch.split(rot_dir, 1, dim=1)
		K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

		zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
		K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
			.view((batch_size, 3, 3))

		ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
		rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
		return rot_mat

	def batch_rigid_transform(self, rot_mats, joints, parents, dtype=torch.float32):
		joints = torch.unsqueeze(joints, dim=-1)

		rel_joints = joints.clone()
		rel_joints[:, 1:] -= joints[:, parents[1:]]

		transforms_mat = self.transform_mat(
			rot_mats.reshape(-1, 3, 3),
			rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

		transform_chain = [transforms_mat[:, 0]]
		for i in range(1, parents.shape[0]):
			# Subtract the joint location at the rest pose
			# No need for rotation, since it's identity when at rest
			curr_res = torch.matmul(transform_chain[parents[i]],
									transforms_mat[:, i])
			transform_chain.append(curr_res)

		transforms = torch.stack(transform_chain, dim=1)

		# The last column of the transformations contains the posed joints
		posed_joints = transforms[:, :, :3, 3]

		joints_homogen = F.pad(joints, [0, 0, 0, 1])

		rel_transforms = transforms - F.pad(
			torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

		return posed_joints, rel_transforms

	def transform_mat(self, R, t):
		''' Creates a batch of transformation matrices
			Args:
				- R: Bx3x3 array of a batch of rotation matrices
				- t: Bx3x1 array of a batch of translation vectors
			Returns:
				- T: Bx4x4 Transformation matrix
		'''
		# No padding left or right, only add an extra row
		return torch.cat([F.pad(R, [0, 0, 0, 1]),
						F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

	def set_params(self, pose=None, beta=None, trans=None, with_body=False):
		"""
		Set pose, shape, and/or translation parameters of SMPL model. Verices of the
		model will be updated and returned.

		Parameters:
		---------
		pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
		relative to parent joint. For root joint it's global orientation.
		Represented in a axis-angle format.

		beta: Parameter for model shape. A vector of shape [10]. Coefficients for
		PCA component. Only 10 components were released by MPI.

		trans: Global translation of shape [3].

		Return:
		------
		Updated vertices.

		"""
		# posed body
		G, B = self.forward(pose, with_body)
		# rest pose body
		G_rest, _ = self.forward(self._rest_pose, with_body=False)
		# from rest to pose
		for i in range(G.shape[0]):
			G[i] = G[i] @ np.linalg.inv(G_rest[i])
		return G, B
		# # posed body
		# G, B = self.update(pose, beta, with_body)
		# # rest pose body
		# G_rest, _ = self.update(self._rest_pose, beta, with_body=False)
		# # from rest to pose
		# for i in range(G.shape[0]):
		# 	G[i] = G[i] @ np.linalg.inv(G_rest[i])
		# return G, B

	def update(self, pose, beta, with_body):
		"""
		Called automatically when parameters are updated.

		"""
		# how beta affect body shape
		v_shaped = self.shapedirs.dot(beta) + self.v_template
		# beta = torch.tensor(beta)
		# v_shaped = torch.einsum('bl,mkl->bmk', [beta[None,:], self.shapedirs]) + self.v_template

		# joints location
		J = self.J_regressor.dot(v_shaped)
		# align root joint with origin
		v_shaped -= J[:1]
		J -= J[:1]
		pose_cube = pose.reshape((-1, 1, 3))
		# rotation matrix for each joint
		R = self.rodrigues(pose_cube)
		# world transformation of each joint
		G = np.empty((self.kintree_table.shape[1], 4, 4))
		G[0] = self.with_zeros(np.hstack((R[0], J[0, :].reshape([3, 1]))))
		for i in range(1, self.kintree_table.shape[1]):
			G[i] = G[self.parent[i]].dot(
				self.with_zeros(
					np.hstack(
						[R[i],((J[i, :]-J[self.parent[i],:]).reshape([3,1]))]
					)
				)
			)
		G = G - self.pack(
			np.matmul(
				G,
				np.hstack([J, np.zeros([24, 1])]).reshape([24, 4, 1])
				)
			)
		v = None
		if with_body:
			I_cube = np.broadcast_to(
				np.expand_dims(np.eye(3), axis=0),
				(R.shape[0]-1, 3, 3)
			)
			lrotmin = (R[1:] - I_cube).ravel()
			# how pose affect body shape in zero pose
			v_posed = v_shaped + self.posedirs.dot(lrotmin)	
			# transformation of each vertex
			T = np.tensordot(self.weights, G, axes=[[1], [0]])
			rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
			v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]

		return G, v

	def rodrigues(self, r):
		"""
		Rodrigues' rotation formula that turns axis-angle vector into rotation
		matrix in a batch-ed manner.

		Parameter:
		----------
		r: Axis-angle rotation vector of shape [batch_size, 1, 3].

		Return:
		-------
		Rotation matrix of shape [batch_size, 3, 3].

		"""
		theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
		# avoid zero divide
		theta = np.maximum(theta, np.finfo(np.float64).eps)
		r_hat = r / theta
		cos = np.cos(theta)
		z_stick = np.zeros(theta.shape[0])
		m = np.dstack([
			z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
			r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
			-r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
		).reshape([-1, 3, 3])
		i_cube = np.broadcast_to(
			np.expand_dims(np.eye(3), axis=0),
			[theta.shape[0], 3, 3]
		)
		A = np.transpose(r_hat, axes=[0, 2, 1])
		B = r_hat
		dot = np.matmul(A, B)
		R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
		return R

	def with_zeros(self, x):
		"""
		Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

		Parameter:
		---------
		x: Matrix to be appended.

		Return:
		------
		Matrix after appending of shape [4,4]

		"""
		return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

	def pack(self, x):
		"""
		Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
		manner.

		Parameter:
		----------
		x: Matrices to be appended of shape [batch_size, 4, 1]

		Return:
		------
		Matrix of shape [batch_size, 4, 4] after appending.

		"""
		return np.dstack((np.zeros((x.shape[0], 4, 3)), x))