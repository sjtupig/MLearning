from mxnet import nd, autograd
from mxnet.gluon import data as gdata, loss as gloss, Trainer as gTrainer, nn

class FMM(nn.Block):
	def __init__(self, n_feature, n_field, f, feature_field_index):
		'''
		n_feature:特征数
		n_field： field数
		f:隐变量长度
		feature_field_index:每个特征对应的field的索引
		'''
		super(FMM, self).__init__()
		self.w = self.params.get('w', shape = (n_feature, 1))
		self.b = self.params.get('b', shape = (1, 1))
		self.latent_vec = self.params.get('latent_vec', shape = (n_feature, n_field, f))
		self.feature_field_index = feature_field_index


	def getBatchData(self, X, y, batchsize = None):
		if batchsize:self.batchsize = batchsize 
		dataset = gdata.ArrayDataset(X, y)
		batchdatas = gdata.DataLoader(dataset, self.batchsize,  shuffle = True)
		return batchdatas

	def forward(self, X):
		#这里不可以使用 += 等inplace操作
		self.linear_item = nd.dot(X, self.w.data())
		self.interaction_item = 0
		for i in range(self.w.shape[0]-1):
			for j in range(i+1, self.w.shape[0]):
				self.interaction_item = self.interaction_item + nd.sum(self.latent_vec.data()[i,self.feature_field_index[j]] * self.latent_vec.data()[j,self.feature_field_index[i]]) * X[:,i].reshape(-1,1) * X[:,j].reshape(-1,1)
		self.y_hat = self.linear_item + self.interaction_item + self.b.data()
		return self.y_hat


	def FMloss(self, y_hat, y, l1 = None, lossf = 'sigmodbinary'):
		loss = gloss.SigmoidBinaryCrossEntropyLoss()
		return loss(y_hat, y) + (0.0 if not l1 else l1*(nd.sum(nd.abs(self.w.data())) + nd.sum(nd.abs(self.b.data())) + nd.sum(nd.abs(self.latent_vec.data()))))


	def train(self, X, y, max_iter = 1000, tol = 1e-5, batchsize = None, l1 = 0.0, l2 = 0.0, verbose = True):
		y_hat = self.forward(X)
		lastloss = self.FMloss(y_hat, y, l1).mean()
		self.batchsize = batchsize if batchsize else int(len(X)/20)
		batchdatas = self.getBatchData(X, y, batchsize)
		self.Trainer = gTrainer(self.params, 'sgd', {'learning_rate':0.03, 'wd':l2})
		for i in range(1, max_iter+1):
			for batchX, batchy in batchdatas:
				with autograd.record():
					batchy_hat = self.forward(batchX)
					l = self.FMloss(batchy_hat, batchy)
				l.backward()
				self.Trainer.step(self.batchsize)
			y_hat = self.forward(X)
			loss = self.FMloss(y_hat, y, l1).mean()
			loss_reduction = lastloss - loss
			lastloss = loss
			if verbose:
				print('%d iter:'%i, 'loss is:', loss, 'loss reduction is:', loss_reduction)

			if loss_reduction < tol: break

	def predict_prob(self, X):
		y_hat = self.forward(X)
		return nd.sigmoid(y_hat)

	def predict(self, X):
		prob = self.predict_prob(X)
		y_hat = prob >= 0.5
		return y_hat
		