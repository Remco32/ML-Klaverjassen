"""
Simpler version of a card game to try out some reinforcement learning
algorithms and mechanics
"""

import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pdb




##VARIABLES
#learning rate and discount rate and epochs
alpha, y, epochs, savingEpoch, printEpoch = 0.01, 0.9, 30000, 3000, 300
#rewards
r1Win, r1Lose, r2Win, r2Lose = 3, -0.5, 2, -1
#to keep track of the total reward obtained by player 2 (who in this simple
#game should always win)
rew1 = []
rew2 = []
#load parameters?
loadP = 1
#to save the weights
FOLDER = '/Users/tommi/github/ML-Klaverjassen/simpler/weights/'
PATH1 = FOLDER + 'net1_weights.pth'
PATH2 = FOLDER + 'net2_weights.pth'





##THE NETWORK
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conn1 = nn.Linear(12,7)  
        self.conn4 = nn.Linear(7,6)
    def forward(self, x):
        x = torch.sigmoid(self.conn1(x))
        x = self.conn4(x)
        return x
    
#net for the two players
net1 = Net()
net2 = Net()
    
#load parameters from previous training rounds
if loadP == 1:
    net1.load_state_dict(torch.load(PATH1))
    net2.load_state_dict(torch.load(PATH2))
    print('Loaded model parameters from folder {}'.format(FOLDER))

#stochastic gradient descent optimiser
opt1 = torch.optim.SGD(net1.parameters(), lr=alpha, momentum=0.9)
opt2 = torch.optim.SGD(net2.parameters(), lr=alpha, momentum=0.9)
loss1 = nn.MSELoss()
loss2 = nn.MSELoss()




#THE GAME
#start counter
start = time.time()

#start the training cycle
for i in range(epochs):

    #re-initialise the feature vectors
    p1 = torch.zeros(12, dtype=torch.float, requires_grad=True)
    p2 = torch.zeros(12, dtype=torch.float, requires_grad=True)
    p1_id, p2_id = [0,2,4], [1,3,5]
    with torch.no_grad():
        for k in range(12):
            if k in p1_id:
                p1[k] = 1
            elif k in p2_id:
                p2[k] = 1
       
    for j in range(3):
        opt1.zero_grad()
        opt2.zero_grad()
        out1 = net1(p1)
        out2 = net2(p2)
        a1 = out1.argmax()
        a2 = out2.argmax()
        
        while a1.item() not in p1_id:     #learn that it's a mistake
            r1 = -1
            rew1.append(r1)
            with torch.no_grad():
                P1 = p1.clone()
                _Q1 = net1(P1)
                Q1 = out1.clone()
            Q1[a1] += alpha * ( r1 + y * torch.max(_Q1).item() - Q1[a1])
            l1 = loss1(Q1, out1)              #square loss
            l1.backward()
            opt1.step()
            opt1.zero_grad()
            out1 = net1(p1)
            a1 = out1.argmax()

        while a2.item() not in p2_id:     #learn that it's a mistake
            r2 = -1
            rew2.append(r2)
            with torch.no_grad():
                P2 = p2.clone()
                _Q2 = net2(P2)
                Q2 = out2.clone()
            Q2[a2] += alpha * ( r2 + y * torch.max(_Q2).item() - Q2[a2])
            l2 = loss2(Q2, out2)              #square loss
            l2.backward()
            opt2.step()
            opt2.zero_grad()
            out2 = net2(p2)
            a2 = out2.argmax()
        
        #Simple game rule: the highest index wins
        if a1.item() > a2.item():
            r1 = r1Win
            r2 = r2Lose 
            rew1.append(r1)
            rew2.append(r2)
        elif a2.item() > a1.item():
            r2 = r2Win
            r1 = r1Lose
            rew1.append(r1)
            rew2.append(r2)
        
        with torch.no_grad():
            P1 = p1.clone()
            P2 = p2.clone()
            P1[a1] = 0
            P2[a2] = 0
            P1[6 + a1] = 1
            P1[6 + a2] = 1
            P2[6 + a1] = 1
            P2[6 + a2] = 1
        
        p1_id.remove(a1.item())
        p2_id.remove(a2.item())
        with torch.no_grad():
            _Q1 = net1(P1)          #next state Q values
            _Q2 = net2(P2)
            Q1  = out1.clone() 
            Q2  = out2.clone()
        Q1[a1] += alpha * ( r1 + y * torch.max(_Q1).item() - Q1[a1])
        Q2[a2] += alpha * ( r2 + y * torch.max(_Q2).item() - Q2[a2])
        l1 = loss1(Q1, out1)
        l2 = loss2(Q2, out2)
        l1.backward()
        l2.backward()
        opt1.step()
        opt2.step()
        p1 = P1.clone()
        p2 = P2.clone()
        #print('Hands after playing:\n{},\n{}'.format(p1,p2))
    if i % printEpoch == 0:
        print('Epoch {} of {}'.format(i,epochs))
    if i  % savingEpoch == 0:
        torch.save(net1.state_dict(), PATH1)
        torch.save(net2.state_dict(), PATH2)
        print('Epoch {} of {}\nSaved weight files in {}'.format(i,epochs,FOLDER))



        
#AFTER THE GAME

#save params
torch.save(net1.state_dict(), PATH1)
torch.save(net2.state_dict(), PATH2)
print('Epoch {} of {}\nSaved weight files in {}'.format(i,epochs,FOLDER))

#calculate total training time
elapsed = time.time() - start
print('\n\nTotal training time: {:.6}'.format(elapsed))

#plot the reward
r1 = np.array(rew1)
r2 = np.array(rew2)
plt.plot(r1,'ob')
plt.plot(r2,'xr')
plt.show()

