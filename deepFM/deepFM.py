from mxnet.gluon import nn, Trainer as gTrainer, loss as gloss
from mxnet import nd, autograd

#FM可以用来做稀疏输入的embedding层

class DeepFM(nn.Block):
	def __init__(self):
		super(self, DeepFM).__init__()
		