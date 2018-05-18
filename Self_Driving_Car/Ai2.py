import numpy as mp
import random as  rp
import os
import torch as T
import torch.nn as N
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

class Network(nn.Module):
# using inheritance here this class is a child of the nn.Module class

    def __init__(self,input_size,nb_action):
        super(Network,self).__init__()
        self.input_size=input_size  # here in our example ther are 5 input var and 3 nb_aution that specifires the output
        self.nb_action=nb_action
        #here we are only creating 1 hidden layer therefore there must be
        # two connections one between input layer and HL 
        # the other between HL and Output Layer 
        self.fc1=N.Linear(input_size,30)
        self.fc2=N.Linear(30,nb_action)
        
    def forward(self,state):
        # activate the hidden neurons
        # represented by x
        x=F.relu(self.fc1(state))
        q_values=self.fc2(x)
        return q_values

class ReplayMemory(object):
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=[]
        
    def push(self,event):
        self.memory.append(event)
        #make sure that memory has capacity events
        # and not more
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self,batch_size):
        samples=zip(*rp.sample(self.memory,batch_size))
        return map(lambda x:Variable(T.cat(x,0)),samples)
    
class Dqn():
    def __init__(self,input_size,nb_action,gamma):
        self.gamma=gamma
        self.reward_window=[]
        self.model=Network(input_size,nb_action)
        self.memory=ReplayMemory(100000)
        self.optimizer=optim.Adam(self.model.parameters(),lr = 0.001)
        self.last_state=T.Tensor(input_size).unsqueeze(0)
        self.last_action=0
        self.last_reward=0
        
    def select_action(self,state):
        probs=F.softmax(self.model(Variable(state,volatile = True))*7)
        #T=7 tempature parameter when T is close to 0 it is less sure
        action = probs.multinomial()
        return action.data[0,0]
     
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
        
    def update(self, reward, new_signal):
        new_state = T.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, T.LongTensor([int(self.last_action)]), T.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        T.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = T.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")