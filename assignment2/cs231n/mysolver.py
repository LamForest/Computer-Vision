# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


class Mysolver(object):

    def __init__(self, model, data, **kwargs):
        #初始化数据
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        #初始化hyperparametr及一些设置
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.learning_decay = kwargs.pop('learning_decay', 0.95)
        self.reg = kwargs.pop('reg', 0)
        self.algorithm = kwargs.pop('algorithm', 'sgd')
        self.batch_size = kwargs.pop('batch_size', 50)
        self.num_train = self.X_train.shape[0]
        self.num_val = self.X_val.shape[0]
        if(self.batch_size > self.num_train):
            self.batch_size = self.num_train

        self.num_epoch = kwargs.pop('num_epochs', 50) #每一个epoch训练num_train / batch_size遍
        self.verbose = kwargs.pop('verbose', True)
        self.print_every = kwargs.pop('print_every', 20)
        self.weight_scale = kwargs.pop('weight_scale', np.sqrt(2.0/self.X_train.shape[0]) ) #当为ReLu时，cs231n所建议的



        #训练数据
        #self.model.params_init(self.weight_scale)

        #初始化后期画图分析时需要的变量
        self.train_acc_history = []
        self.val_acc_history = []
        self.loss_history = []




    '''
    @brief bp一次
    @note 抽出self.batchsize个数的数据，调用model的一次loss函数，获得grads,
    loss，更新model.params，然后将loss放入loss_history中。
    @return 无
    '''
    def _step(self):
        #选不重复的
        batch_mask = np.random.choice(self.num_train, self.batch_size, False)
        batch_X = self.X_train[batch_mask]
        batch_y = self.y_train[batch_mask]

        #BP
        loss, grads = self.model.loss(batch_X, batch_y)
        self.loss_history.append(loss)

        '''
        @brief 更新params
        @note 此处有多种方式
        比如k,v in dict.items()将词典{'age': 18, 'name': 'gao'}
        变为了[('age', 18), ('name', 'gao')]这样一个数组，然后就可以for in了
        但是当有很多层网络时，这样会产生非常大的内存占用，所以采用iteritems，
        这是最快最好的方式，因为它产生了一个生成器，类似xrange
        '''
        for k in self.model.params.iterkeys():
            self.model.params[k] -= self.learning_rate * grads[k]

    '''
    @brief 统计某个集合的准确度
    @note 为什么要把这个功能写成一个函数呢。首先，因为这个功能被两个地方用到，训练集和测试集。
    其次，在测试准确度的时候，最好的方式是分批测试，而不是49000个测试集全部传进去，会产生大量的中间结果
    非常占用内存
    @todo 改为分批处理
    @return 准确度
    '''
    def check_accuracy(self, X, y):
        '''
        如何统计两个数组中相同元素的个数呢？
        np.mean轻松搞定
        '''
        scores = self.model.loss(X) # train_scores N * C
        acc = np.mean(y == np.argmax(scores, axis = 1))
        return acc



    def train(self):
        num_iter_per_epoch = self.num_train / self.batch_size
        iter_sum_times = num_iter_per_epoch * self.num_epoch
        iter_times = 0;
        for num_epoch in xrange(self.num_epoch):

            for num_iter in xrange(num_iter_per_epoch):
                #BP一次
                self._step()

                if(iter_times % self.print_every == 0):
                    print ('Epoch %d, iter_times %d/%d -> loss %f') % (num_epoch, iter_times,iter_sum_times, self.loss_history[iter_times])

                iter_times += 1;

            '''
            打印Accuracy,每一个epoch计算并打印一次。
            '''
            train_acc = self.check_accuracy(self.X_train, self.y_train)
            self.train_acc_history.append(train_acc)
            val_acc = self.check_accuracy(self.X_val, self.y_val)
            self.val_acc_history.append(val_acc)
            print ("Epoch %d, Iteration times: %d/%d, train_acc: %f, val_acc: %f") % (num_epoch, iter_times, iter_sum_times, train_acc, val_acc)

        # self.iter_times = iter_times

    def plot(self):
        plt.subplot(2,1,1)
        plt.title('Train Loss')
        plt.plot(self.loss_history, 'o')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        plt.subplot(2,1,2)
        plt.title('Accuracy')
        plt.plot(self.train_acc_history, '-o', label='train')
        plt.plot(self.val_acc_history, '-o', label='validation')
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.gcf().set_size_inches(15,12) # ???
        plt.show()
