"""
Simpler version of a card game to try out some reinforcement learning
algorithms and mechanics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#learning rate and discount rate
alpha, y = 0.01, 0.9

#the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conn1 = nn.Linear(8,20)  #input - first HL
        self.conn2 = nn.Linear(20,4)  #first HL - output
    def forward(self, x):
        x = torch.sigmoid(self.conn1(x))
        x = self.conn2(x)
        return x

    
#net for the two players
net1 = Net()
net2 = Net()

#stochastic gradient descent optimiser
opt1 = torch.optim.SGD(net1.parameters(), lr=alpha, momentum=0.9)
opt2 = torch.optim.SGD(net2.parameters(), lr=alpha, momentum=0.9)

 #two tensors represent the player's feature vectors
p1_list, p2_list, table_list = [1,0,1,0], [0,1,0,1], [0,0,0,0]
p1 = torch.tensor(p1_list + table_list, dtype=torch.float, requires_grad=True)
p2 = torch.tensor(p2_list + table_list, dtype=torch.float, requires_grad=True)
Q1 = torch.zeros(4)
Q2 = torch.zeros(4)


for i in range(500):    
    p1_id, p2_id = [0,2], [1,3]
    with torch.no_grad():
       for i in range(8):
           p1[i] = 0
           p2[i] = 0
           if i in p1_id:
               p1[i] = 1
           if i in p2_id:
                p2[i] == 1
        
    
    for j in range(2):
        #e.g. [1,0,1,0] --> either [0,0,1,0] or [1,0,0,0]
        print(p1, p2)
        opt1.zero_grad()
        opt2.zero_grad()
        out1 = net1(p1)
        out2 = net2(p2)
        a1 = out1.argmax()
        a2 = out2.argmax()

        while a1 not in p1_id:     #learn that it's a mistake
            r1 = -10
            _Q1 = net1(p1)
            Q1 += alpha * ( r1 + y * torch.max(_Q1).item() - Q1)   #QLearning rule
            loss1 = ( 0.5 * ( _Q1 - Q1 ) ** 2 ).sum()              #square loss
            opt1.zero_grad()
            loss1.backward()
            opt1.step()
            out1 = net1(p1)
            a1 = out1.argmax()

        while a2 not in p2_id:
            r2 = -10
            _Q2 = net2(p2)
            Q2 += alpha * ( r2 + y * torch.max(_Q2).item() - Q2)
            loss2 = ( 0.5 * ( _Q2 - Q2 ) ** 2 ).sum()
            opt2.zero_grad()
            loss2.backward()
            opt2.step()
            out2 = net2(p2)
            a2 = out2.argmax()
        
        #Simple game rule: the highest index wins
        if a1 > a2:
            r1 = 10
            r2 = -1           #not as bad as playing the wrong card
        elif a2 > a1:
            r2 = 10
            r1 = -1

        with torch.no_grad():     
            p1[a1] = 0
            p2[a2] = 0
            p1[4 + a1] = 1
            p1[4 + a2] = 1
            p2[4 + a1] = 1
            p2[4 + a2] = 1

        p1_id.remove(a1)
        p2_id.remove(a2)
        _Q1 = net1(p1)          #next state Q values
        _Q2 = net2(p2)
        Q1[a1] += alpha * ( r1 + y * torch.max(_Q1).item() - Q1[a1])
        Q2[a2] += alpha * ( r2 + y * torch.max(_Q2).item() - Q2[a2])
        loss1 = ( 0.5 * ( _Q1 - Q1 ) ** 2 ).sum()
        loss2 = ( 0.5 * ( _Q2 - Q2 ) ** 2 ).sum()
        opt1.zero_grad()
        loss1.backward()
        opt2.zero_grad()
        loss2.backward()
        opt1.step()
        opt2.step()

