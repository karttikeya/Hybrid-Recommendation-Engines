import numpy as np
import pandas as pd
import math
from scipy.stats import logistic
import pickle
import sys
from random import shuffle

class RBM :
    def __init__(self, numhids, items, users, transactions):
        self.numdims = len(items)
        self.numhids = numhids
        self.epsilonw = 0.1   # learning rate for weights
        self.epsilonvb = 0.1;  # learning rate for biases for visible units
        self.epsilonhb = 0.1;  # learning rate for biases for hidden units
        self.weightcost = 0.0002;
        self.weightcost = 0.002;
        self.initialmomentum = 0.5;
        self.finalmomentum = 0.9;
        self.modifier = 20.0;
        self.nclasses = 10;    
        self.eta = 0.1;
        self.momentum = 0.5; 
        self.maxepoch = 30;
        self.avglast = 0;
        self.penalty = 2e-4;
        self.batchsize = 500;
        self.verbose = True;
        self.anneal = False;    
        self.numhid = 500;
        self.debug = True;
        self.restart = True;
        self.createSnapshotEvery = 100;
        self.nummachines = len(users)
        self.five = max(transactions['rating'])
        print self.five
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
        
        
        self.Wijk = np.random.normal(size=(self.five, self.numdims, self.numhids))*(0.1)
        self.hidbiaises = np.zeros(self.numhids)
        self.visbiaises = np.zeros([self.five, self.numdims])
        self.matrix = [[] for i in range(self.nummachines)]
        for i in range(len(transactions)) :
            self.matrix[self.u2h[transactions['userId'][i]]].append([self.i2h[transactions['itemId'][i]] , transactions['rating'][i]])
        
        self.R = np.zeros([self.nummachines, self.numdims])
        for i in range(len(transactions)) :
            self.R[self.u2h[transactions['userId'][i]]][self.i2h[transactions['itemId'][i]]] = 1
        self.answer = [[] for i in range(self.nummachines)]
        self.score = np.zeros([self.nummachines, self.numdims])

    def train(self) :
        
        startAveraging = self.maxepoch - self.avglast


        Wijk_inc = np.zeros([self.five, self.numdims, self.numhids])
        posprods = np.zeros([self.five, self.numdims, self.numhids])
        negprods = np.zeros([self.five, self.numdims, self.numhids])
        hidbiais_inc = np.zeros(self.numhids)
        visbiais_inc = np.zeros([self.five, self.numdims])
        negdata = np.zeros([self.five,self.numdims])
        V = np.zeros([self.five,self.numdims])

        for epoch in range(1,self.maxepoch+1) :

            print "Starting epoch ", epoch
            errsum=0.0
            poshidact = np.zeros(self.numhids)
            neghidact = np.zeros(self.numhids)
            posvisact = np.zeros([self.five,self.numdims])
            negvisact = np.zeros([self.five,self.numdims])

            #separate loop for each RBM.
            NM = range(self.nummachines)
            shuffle(NM)
            for r in NM:
                row = self.matrix[r]
                for i in range(len(row)) :
                    posvisact[row[i][1]-1][row[i][0]] += 1

                poshidprobs = np.zeros(self.numhids)
                poshidprobs += self.hidbiaises
                for i in range(len(row)) :
                    poshidprobs += self.Wijk[row[i][1]-1][row[i][0]]
                poshidprobs = logistic.cdf(poshidprobs)
                poshidact += poshidprobs

                for i in range(len(row)) :
                    posprods[row[i][1]-1][row[i][0]] += poshidprobs

                #end of positive phase
                poshidstates = poshidprobs > np.random.rand(self.numhids)
                #print 'poshidstates', np.mean(poshidstates)

                for i in range(len(row)) :
                    negdata[:,row[i][0]] = np.sum(np.transpose(self.Wijk[:,row[i][0]]) * poshidstates[:,np.newaxis] , axis=0)
                    negdata[:,row[i][0]] += self.visbiaises[:,row[i][0]]
                    negdata[:,row[i][0]] = np.exp(negdata[:,row[i][0]])
                    
                    sum1 = np.sum(negdata[:,row[i][0]])
                    if sum1==0 : sum1=1
                    negdata[:,row[i][0]] /= sum1
                for i in range(len(row)) :
                    negvisact[:,row[i][0]] += negdata[:,row[i][0]]

                neghidprobs = np.zeros(self.numhids)
                neghidprobs += self.hidbiaises
                
                for i in range(len(row)) :
                    neghidprobs += np.sum(self.Wijk[:,row[i][0]] * negdata[:,row[i][0]][:,np.newaxis], axis=0)
                neghidprobs = logistic.cdf(neghidprobs)
                #print 'neghidprobs: ', np.mean(neghidprobs)
                neghidact += neghidprobs

                for i in range(len(row)) :
                    negprods[:,row[i][0]] += np.outer(negdata[:,row[i][0]] , np.transpose(neghidprobs))

                err=0.0
                for i in range(len(row)) :
                    V[row[i][1]-1][row[i][0]] = 1
                    err += np.sum((negdata[:,row[i][0]] - V[:,row[i][0]])**2)
                err = math.sqrt(err)
                errsum += err

                #set momentum
                momentum = 0.0
                if epoch > startAveraging :
                    momentum = self.finalmomentum
                else :
                    momentum = self.initialmomentum

                for i in range(len(row)) :
                    V[:,row[i][0]] = 0
                    negdata[:,row[i][0]] = 0

            Wijk_inc = Wijk_inc*momentum + (posprods-negprods)*(self.epsilonw/(self.nummachines+0.0)) - self.Wijk*self.weightcost
            visbiais_inc = momentum*visbiais_inc + (self.epsilonvb/self.nummachines)*(posvisact-negvisact)
            hidbiais_inc = momentum*hidbiais_inc + (self.epsilonhb/self.nummachines)*(poshidact-neghidact)
                #update connection weights
            self.Wijk += Wijk_inc    
            self.hidbiaises += hidbiais_inc
            self.visbiaises += visbiais_inc

            print "Epoch " , epoch , " error " , errsum , "\n"

        self.pre_compute_recommendations()

    def pre_compute_recommendations(self) :

        for i in range(self.nummachines) :
			self.answer[i] = self.precompute_single_recommendations(i, self.numdims - len(self.matrix[i]))

        with open('answer.pickle', 'wb') as handle:
            pickle.dump(self.answer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('score.pickle', 'wb') as handle:
            pickle.dump(self.score, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def precompute_single_recommendations(self, user, N) :

        user_ratings_so_far = self.matrix[user]
        #positive phase
        poshidprobs = np.zeros(self.numhids)
        poshidprobs += self.hidbiaises

        for i in range(len(user_ratings_so_far)) :
            poshidprobs += self.Wijk[user_ratings_so_far[i][1]-1][user_ratings_so_far[i][0]]
        poshidprobs = logistic.cdf(poshidprobs)
        poshidstates = poshidprobs > np.random.rand(self.numhids)
        print np.mean(poshidstates)

        #start negative phase
        negdata = np.zeros([self.five, self.numdims])
        for tmp in range(self.five) :
            negdata[tmp] = np.dot(self.Wijk[tmp], poshidstates)
            negdata[tmp] += self.visbiaises[tmp]
        negdata = np.exp(negdata)
        sum1 = np.sum(negdata,axis=0)
        negdata /= sum1
        tmp = np.zeros([self.five, self.numdims])
        for i in range(self.five) :
            tmp[i] = i+1
        score = np.sum(negdata*tmp , axis=0)
        self.score[user] = score
        print np.mean(score)
        for en in user_ratings_so_far :
        	score[en[0]] = -1

        return list(np.argsort(score)[-N:])
    
    def evaluate_RBM(self, test_data) :
        
        err=0.0
        for i in range(len(test_data)) :
        	user = self.u2h[test_data['userId'][i]]
        	item = self.i2h[test_data['itemId'][i]]
        	rating = test_data['rating'][i]
        	pred = self.score[user][item]
        	err += abs(rating-pred)
        print (0.0+err)/len(test_data)
    

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
    
    r = RBM(100, items, users, train_transactions)
    r.train()
    
    test_data = pd.read_csv('test_ratings.csv')
    
    r.evaluate_RBM(test_data)
    print r.recommend_users([1,30], 10)