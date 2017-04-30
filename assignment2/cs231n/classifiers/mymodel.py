# -*- coding:utf-8 -*-
import numpy as np

class MyAwesomeModel(object):
    '''
    @brief 初始化architecture，并不初始化W、b
    '''
    def __init__(self, input_dim = 3*32*32, num_class = 10, hidden_dims = [10, 10, 10]):
        self.params = {}

        self.num_layers = len(hidden_dims) + 1
        hidden_dims = [input_dim] + hidden_dims + [num_class]
        self.hidden_dims = hidden_dims



    '''
    @brief  weight_scale = 1e-3,
    @note 参数的初始化并不在__init__中做，因为生成model时，只是给定了model的architecture，
    而只有在solver中才能确定weight_scale，所以两个工作要分开
    '''
    def params_init(self,  weight_scale = 1e-3 ):
        for i in xrange(self.num_layers):
            self.params['W' + str(i+1)] = weight_scale * np.random.rand(self.hidden_dims[i], self.hidden_dims[i+1])
            self.params['b' + str(i+1)] = np.zeros(hidden_dims[i+1])

    def loss(self, X, y=None):
        
