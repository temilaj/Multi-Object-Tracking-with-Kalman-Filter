import numpy as np

class KalmanFilter(object):
	def __init__(self):
		"""
		P = previous covariance matrix
		F = state transition matrix
		Q = process noise matrix
		"""
		self.dt = 0.005 

		self.A = np.array([[1, 0], [0, 1]]) 
		self.u = np.zeros((2, 1))  
		self.P = np.diag((3.0, 3.0)) 
		self.F = np.array([[1.0, self.dt], [0.0, 1.0]]) 

		self.Q = np.eye(self.u.shape[0]) 

		self.b = np.array([[0], [255]]) 

		self.R = np.eye(self.b.shape[0]) 
		self.previous_result = np.array([[0], [255]])

	def predict_state_vector(self):
	   
		self.u = np.round(np.dot(self.F, self.u))
		self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
		self.previous_result = self.u 
		return self.u

	def correct_state_vector(self, b, flag):
		if not flag:
			self.b = self.previous_result
		else:
			self.b = b
		C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R
		K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))

		self.u = np.round(self.u + np.dot(K, (self.b - np.dot(self.A,
															  self.u))))
		self.P = self.P - np.dot(K, np.dot(C, K.T))
		self.previous_result = self.u
		return self.u
