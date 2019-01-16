from mxnet.gluon import nn, Trainer as gTrainer, loss as gloss, data as gdata
from mxnet import nd, autograd
import numpy as np 

#FM可以用来做稀疏输入的embedding层

class DeepFM(nn.Block):
	def __init__(self,n_features, f, field_num):
		super(DeepFM, self).__init__()
		if not (f > 0 and isinstance(f,int)):
			raise BaseException('f must be larger than 0')
		#embeddings
		self.field_num = field_num
		self.f = f 
		self.latent_vec = self.params.get('latent_vec', shape=(n_features, f))

		#FM, fisrt order
		self.w = self.params.get('w', shape=(n_features, 1))
		self.b = self.params.get('b', shape=(1, 1))
		

		#DNN
		self.dense1 = nn.Dense(200, activation='relu')
		self.dense2 = nn.Dense(200, activation='relu')
		self.dense3 = nn.Dense(1)

	def forward(self, X):
		#FM first order
		self.FM_FirtOrder = nd.dot(X,self.w.data()) + self.b.data()

		#FM second order, reference FM.py
		self.FM_SecondOrder = nd.sum(nd.power(nd.dot(X, self.latent_vec.data()),2) - nd.dot(nd.power(X,2),nd.power(self.latent_vec.data(),2)), axis = 1, keepdims=True)

		#DNN
		#get embeddings layer
		self.embeddings = nd.take(self.latent_vec.data(), nd.array(np.argwhere(X.asnumpy())[:,1].reshape(-1,2))).reshape(-1, self.f*self.field_num)
		self.DNN = self.dense3(self.dense2(self.dense1(self.embeddings)))
		self.y_hat = self.FM_FirtOrder + self.FM_SecondOrder + self.DNN
		return self.y_hat

	def getBatchData(self, X, y, batchsize = None):
		if batchsize:self.batchsize = batchsize 
		dataset = gdata.ArrayDataset(X, y)
		batchdatas = gdata.DataLoader(dataset, self.batchsize,  shuffle = True)
		return batchdatas

	def FMloss(self, y_hat, y ):
		loss = gloss.SigmoidBinaryCrossEntropyLoss()
		return loss(y_hat, y)

	def train(self, X, y, max_iter = 1000, tol = 1e-6, batchsize = None, optname = 'ftrl',optparams = {'lamda1':0.01, 'learning_rate':0.1, 'wd':0.01}, verbose = True):
		y_hat = self.forward(X)
		lastloss = self.FMloss(y_hat, y).mean()
		self.batchsize = batchsize if batchsize else int(len(X)/20)
		batchdatas = self.getBatchData(X, y, batchsize)
		self.Trainer = gTrainer(self.collect_params(), optname, optparams)
		for i in range(1, max_iter+1):
			for batchX, batchy in batchdatas:
				with autograd.record():
					batchy_hat = self.forward(batchX)
					l = self.FMloss(batchy_hat, batchy)
				l.backward()
				self.Trainer.step(self.batchsize)
			y_hat = self.forward(X)
			loss = self.FMloss(y_hat, y).mean()
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

    