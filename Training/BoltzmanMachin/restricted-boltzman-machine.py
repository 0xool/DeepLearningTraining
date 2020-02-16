
#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim 
import torch.utils.data
from torch.autograd import Variable

# %%
movies = pd.read_csv('ml-1m/movies.dat', '::',header=None,engine='python',encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', '::',header=None,engine='python',encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', '::',header=None,engine='python',encoding='latin-1')

# %%
training_set = pd.read_csv('ml-100k/u1.base',delimiter='\t')
training_set = np.array(training_set,dtype='int')
test_set = pd.read_csv('ml-100k/u1.test',delimiter='\t')
test_set = np.array(test_set,dtype='int')

# %%
maximum_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
maximum_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

# %%
def convert(data):
    new_data = []
    for u in range(1,maximum_users + 1):
        id_movies = data[:,1][data[:,0] == u]
        id_ratings = data[:,2][data[:,0] == u]
        r = np.zeros(maximum_movies)
        r[id_movies - 1] = id_ratings
        new_data.append(list(r))
    return new_data
    
training_set = convert(training_set)
test_set = convert(test_set)
# %%
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)



# %%
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# %%
class RBM():

    def __init__(self,nv,nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)

    def sample_h(self, x):
        wx = torch.mm(x,self.W.t())
        activation = wx + self.a.expand_as(wx)
        P_h_given_v = torch.sigmoid(activation)
        return P_h_given_v,torch.bernoulli(P_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y,self.W)
        activation = wy + self.b.expand_as(wy)
        P_v_given_h = torch.sigmoid(activation)
        return P_v_given_h,torch.bernoulli(P_v_given_h)
    
    def train(self, v0, vk,ph0,phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk) , 0)  
        self.a += torch.sum((ph0 - phk),0)
        pass




    pass

# %%
nv = maximum_movies
nh = 100
batch_size = 100
rbm = RBM(nv=nv ,nh=nh)
#%%
nb_epoch = 10
for epoch in range(1,nb_epoch + 1):
    train_loss = 0
    counter = 0.
    for i in range(0,maximum_users - batch_size, batch_size):
        vk = training_set[i:i + batch_size]
        v0 = training_set[i:i + batch_size]
        ph0,_ = rbm.sample_h(v0)
        
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0<0]
        
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        counter += 1.
        print('epochs : ' + str(epoch))
        print('train loss: ' + str(train_loss/counter))



# %%
test_loss = 0
counter = 0.
for i in range(0,maximum_users):
    v = training_set[i:i + 1]
    vt = test_set[i:i + 1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        counter += 1.
    print('test loss: ' + str(test_loss/counter))



# %%
