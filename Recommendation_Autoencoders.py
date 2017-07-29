import numpy as np
import pandas as pd
import math
import numpy
numpy.random.seed(0)
import CDAE
import metrics
import pickle

class AutoEncoder :

    def __init__(self, items, users, train_transactions, test_transactions):
        
        self.u2h,self.i2h={},{}
        self.h2u,self.h2i={},{}
        for i in range(len(users)) :
            self.u2h[users['userId'][i]] = i
            self.h2u[i] = users['userId'][i]
        for i in range(len(items)) :
            self.i2h[items['itemId'][i]] = i
            self.h2i[i] = items['itemId'][i]

        self.n_users = len(users)
        self.n_items = len(items)
        self.train_users = list(self.h2u.keys())
        self.train_x = np.zeros([self.n_users, self.n_items])
        self.test_users = list(self.h2u.keys())
        self.test_x = np.zeros([self.n_users, self.n_items])
        for i in range(len(train_transactions)) :
            self.train_x[self.u2h[train_transactions['userId'][i]]][self.i2h[train_transactions['itemId'][i]]] = 1
        for i in range(len(test_transactions)) :
            self.test_x[self.u2h[test_transactions['userId'][i]]][self.i2h[test_transactions['itemId'][i]]] = 1


    def train(self) :

        print 'Starting training....'
        
        train_x_users = numpy.array(self.train_users, dtype=numpy.int32).reshape(len(self.train_users), 1)
        test_x_users = numpy.array(self.test_users, dtype=numpy.int32).reshape(len(self.test_users), 1)

        # model
        model = CDAE.create(I=self.train_x.shape[1], U=len(self.train_users)+1, K=50,
                            hidden_activation='relu', output_activation='sigmoid', q=0.50, l=0.01)
        model.compile(loss='mean_absolute_error', optimizer='adam')
        model.summary()

        # train
        history = model.fit(x=[self.train_x, train_x_users], y=self.train_x,
                            batch_size=128, nb_epoch=250, verbose=1,
                            validation_data=[[self.test_x, test_x_users], self.test_x])

        self.pre_compute_recommendations(model)

    def pre_compute_recommendations(self, model) :

        self.answer = model.predict(x=[self.train_x, numpy.array(self.train_users, dtype=numpy.int32).reshape(len(self.train_users), 1)])
        self.answer = self.answer * (self.train_x == 0) # remove watched items from predictions
        self.answer = numpy.fliplr(numpy.argsort(self.answer))

        # with open('answer_case5.pickle', 'wb') as handle:
        #     pickle.dump(self.answer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('answer_case5.pickle', 'rb') as handle :
        #     self.answer = pickle.load(handle)
        

    def evaluate_AutoEncoder(self, N) :

        print 'Starting evaluation....'
        sr = metrics.success_rate(self.answer[:, :N], self.test_x)
        print("Success Rate at {:d}: {:f} %".format(N, sr))

    def recommend_users(self, usersList, N) :

        answer = {}
        for user in usersList :
            tmp = self.answer[self.u2h[user]][:N]
            print len(tmp)
            for i in range(N) :
                tmp[i] = self.h2i[tmp[i]]
            answer[user] = tmp
        return answer

    
if __name__ == '__main__':
    
    users = pd.read_csv('user_data.csv')
    items = pd.read_csv('item_data.csv')
    train_transactions = pd.read_csv('train_ratings.csv')
    test_transactions = pd.read_csv('test_ratings.csv')
    
    aec = AutoEncoder(items, users, train_transactions, test_transactions)
    aec.train()

    aec.evaluate_AutoEncoder(5)
    print aec.recommend_users([10,30],10)