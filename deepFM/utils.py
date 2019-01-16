import numpy as np 
from mxnet import nd 

def labelencoder(nparray):
	try:
		arr = np.array(ndarray)
	except:
		raise BaseException('input must be a list or numpy.array')

	unique_value = np.unique(arr)
	value_index = {ch, i for i, ch in enumerate(unique_value)}
	le = [value_index[i] for i in arr]
	return nd.array(le).reshape(-1,1)