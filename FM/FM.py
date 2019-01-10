from mxnet import nd, autograd, gluon 
from mxnet.gluon import nn, loss as gloss, Trainer as gTrainer, data as gdata


class FM(nn.Block):
    def __init__(self, n_features, f):
    	super(FM, self).__init__()
    	if not (f > 0 and isinstance(f,int)):
    		raise BaseException('f must be larger than 0')
    	self.w = self.params.get('w', shape=(n_features, 1))
    	self.b = self.params.get('b', shape=(1, 1))
    	self.latent_vec = self.params.get('latent_vec', shape=(n_features, f))
    	#self.w = nd.random.normal(loc = 0, scale = 0.01, shape = (X.shape[1], 1))
    	#self.b = nd.random.normal(loc = 0, scale = 0.01, shape = (1, 1))
    	#self.latent_vec = nd.random.normal(loc = 0, scale = 0.01, shape = (X.shape[1], f))

    def getBatchData(self, X, y, batchsize = None):
    	if batchsize:self.batchsize = batchsize 
    	dataset = gdata.ArrayDataset(X, y)
    	batchdatas = gdata.DataLoader(dataset, self.batchsize,  shuffle = True)
    	
    	return batchdatas

    def FMloss(self, y_hat, y, l1 = None, lossf = 'sigmodbinary'):
    	loss = gloss.SigmoidBinaryCrossEntropyLoss()
    	return loss(y_hat, y) + (0.0 if not l1 else l1*(nd.sum(nd.abs(self.w.data())) + nd.sum(nd.abs(self.b.data())) + nd.sum(nd.abs(self.latent_vec.data()))))

    

    def forward(self, X):
    	self.linear_item = nd.dot(X, self.w.data())
    	self.interaction_item = nd.sum(nd.power(nd.dot(X, self.latent_vec.data()),2) - nd.dot(nd.power(X,2),nd.power(self.latent_vec.data(),2)), axis = 1, keepdims=True)
    	self.y_hat = self.linear_item + self.interaction_item + self.b.data()
    	return self.y_hat

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