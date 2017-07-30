import numpy as np
import pandas as pd
import sys
import math
import copy
import scipy.optimize
import CouplingSVD

class Recommender :
    
    def __init__(self, items, users, transactions) :

        self.u2h = {}
        i=0
        for uid in users['userId'] :
            self.u2h[uid]=i
            i+=1

        self.i2h = {}
        i=0
        for iid in items['itemId'] :
            self.i2h[iid]=i
            i+=1

        self.h2u = {}
        i=0
        for uid in users['userId'] :
            self.h2u[i]=uid
            i+=1

        self.h2i = {}
        i=0
        for iid in items['itemId'] :
            self.h2i[i]=iid
            i+=1
            
        self.n_features_users = len(users.columns)-1
        self.n_items = len(items)
        self.n_users = len(users)

        self.UF = np.zeros([self.n_users, self.n_features_users])

        for i in range(self.n_users) :
            j=0
            for col in users.columns :
                if col!='userId' :
                    self.UF[i][j] = users[col][i]
                    j+=1

        self.R = np.zeros([self.n_users,self.n_items])
        self.R1 = [[] for i in range(self.n_users)]
        for i in range(len(transactions)) :
            self.R[self.u2h[transactions['userId'][i]]][self.i2h[transactions['itemId'][i]]] = transactions['rating'][i]
            self.R1[self.u2h[transactions['userId'][i]]].append([self.i2h[transactions['itemId'][i]], transactions['rating'][i]])
        self.UU = np.zeros([self.n_users,self.n_users])
        self.nR = np.zeros_like(self.R)
        self.answer = np.zeros_like(self.R)

    def train(self) :

        for i in range(self.n_users) :
            for j in range(i+1) :

                den = np.linalg.norm(self.UF[i]) * np.linalg.norm(self.UF[j])
                if den==0 :
                    self.UU[i][j] = 0
                else :
                    self.UU[i][j] = (np.sum(self.UF[i]*self.UF[j]))/(den+0.0)
                self.UU[j][i] = self.UU[i][j]

        nP,nQ = CouplingSVD.matrix_factorization_user(self.R1, 100, self.UU, 100, 0.005, 0.1, 0.1, self.n_users, self.n_items)
        self.nR = np.dot(nP, nQ.T)

        self.answer = self.nR * (self.R == 0)
        self.answer = np.fliplr(np.argsort(self.answer))

    def evaluate_Recommender(self, val_data) :

        err=0.0
        for i in range(len(val_data)) :
            user = self.u2h[val_data['userId'][i]]
            item = self.i2h[val_data['itemId'][i]]
            rating = val_data['rating'][i]
            pred = self.nR[user][item]
            err += abs(rating-pred)
        print (0.0+err)/len(val_data)

    def recommend_users(self, usersList, N) :

        answer = {}
        for user in usersList :
            tmp = self.answer[self.u2h[user]][:N]
            for i in range(N) :
                tmp[i] = self.h2i[tmp[i]]
            answer[user] = tmp
        return answer

def pre_process(data, preserved) :
    data2 = copy.deepcopy(data)
    print data2.dtypes
    print data2.columns
    for col in data.columns :
        if col!=preserved :
            if (data.dtypes[col] == 'int64' or data.dtypes[col] == 'float64') and len(data[col].unique()) >= 20:
                x1 = data[col].min()
                x2 = data[col].max()
                data2[col] = (data2[col] - x1)/(x2-x1)
            else :
                cnt = len(data[col].unique())
                print cnt
                if cnt>50 :
                    del data2[col]
                else :
                    data2 = pd.get_dummies(data2, columns = [col])
    print data2.head()
    return data2


if __name__ == '__main__':
    
    users = pd.read_csv('user_data.csv')
    items = pd.read_csv('item_data.csv')
    transactions = pd.read_csv('train_ratings.csv')
    
    #users = pre_process(users, 'userId')
    #items = pre_process(items, 'itemId')

    r = Recommender(items, users, transactions)
    r.train()
    val_data = pd.read_csv('test_ratings.csv')
    r.evaluate_Recommender(val_data)

    r.recommend_users([10,30],10)



