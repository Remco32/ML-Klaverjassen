import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pdb





#VARIABLES
alpha, y, epoch, savingEpoch, printEpoch = 0.01, 0.9, 10000, 30000, 600
00
rew1, rew2 = [], []
r1Win, r1Lose, r2Win, r2Lose = 2, -0.3, 2, -0.3

#load parameters?
loadP = 1
#to save the weights
FOLDER = '/Users/tommi/github/ML-Klaverjassen/simpler/weights/'
PATH1 = FOLDER + 'net1_weights.pth'
PATH2 = FOLDER + 'net2_weights.pth'






#THE NETWORK
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conn1 = nn.Linear(13,20)
        self.conn2 = nn.Linear(20,6)
    def forward(self, x):
        x = torch.sigmoid(self.conn1(x))
        x = self.conn2(x)
        return x

#nets for the players   
n1 = Net()
n2 = Net()

#load parameters from previous training rounds
if loadP == 1:
    n1.load_state_dict(torch.load(PATH1))
    n2.load_state_dict(torch.load(PATH2))
    print('Loaded model parameters from {}'.format(FOLDER))

#optimisers and losses
opt1  = torch.optim.SGD(n1.parameters(), lr = alpha)
opt2  = torch.optim.SGD(n2.parameters(), lr = alpha)
loss1 = nn.MSELoss()
loss2 = nn.MSELoss()






#THE GAME
start = time.time()
for i in range(epoch):

    id1, id2 = [0, 2, 4], [1, 3, 5]
    p1 = torch.zeros(13, dtype=torch.float, requires_grad=True)
    p2 = torch.zeros(13, dtype=torch.float, requires_grad=True)
    with torch.no_grad():
        for k in range(12):
            if k in id1:
                p1[k] = 1
            elif k in id2:
                p2[k] = 1

    for j in range(3):
        opt1.zero_grad()
        out1 = n1(p1)
        a1   = out1.argmax().item()
       
        while a1 not in id1:
            r1 = -1
            rew1.append(r1)
            with torch.no_grad():
                Q1 = out1.clone()
                Q1[a1] += alpha * ( r1 + y * torch.max(out1).item() - Q1[a1])
            l1 = loss1(Q1, out1)
            l1.backward()
            opt1.step()
            opt1.zero_grad()
            out1 = n1(p1)
            a1   = out1.argmax().item()

       # print('\n',j,a1)
       # tmp = input('--continue?')
        with torch.no_grad():
            P2 = p2.clone()
            P2[6] = a1     #to keep track of the played card
            p2 = P2.clone()
            p2.requires_grad = True
            
        opt2.zero_grad()
        out2 = n2(p2)
        a2   = out2.argmax().item()

        while a2 not in id2:
            r2 = -1
            rew2.append(r2)
            with torch.no_grad():
                Q2 = out2.clone()
                Q2[a2] += alpha * ( r2 + y * torch.max(out2).item() - Q2[a2])
            l2 = loss2(Q2, out2)
            l2.backward()
            opt2.step()
            opt2.zero_grad()
            out2 = n2(p2)
            a2   = out2.argmax().item()            
       # print(a2)
       # tmp = input('--continue?')
        if a1 > a2:
            r1 = r1Win
            r2 = r2Lose
            rew1.append(r1)
            rew2.append(r2)
        #    print('Winner: p1')
         #   tmp = input('--continue?')
        if a2 > a1:
            r2 = r2Win
            r1 = r1Lose
            rew1.append(r1)
            rew2.append(r2)
          #  print('Winner: p2')
           # tmp = input('--continue?')

        with torch.no_grad():
            P1 = p1.clone()
            P2 = p2.clone()
            P1[a1] = 0
            P2[a2] = 0
            P1[7 + a2] = 2       #major change
            P1[7 + a1] = 1
            P2[7 + a2] = 2       #major change
            P2[7 + a1] = 1
            
        id1.remove(a1)
        id2.remove(a2)
        with torch.no_grad():
            Q1 = out1.clone()
            Q2 = out2.clone()
        q1 = n1(P1)
        q2 = n2(P2)
        Q1[a1] += alpha * ( y * torch.max(q1).item() - Q1[a1])
        Q2[a2] += alpha * ( y * torch.max(q2).item() - Q2[a2])
        l1 = loss1(Q1, out1)
        l2 = loss2(Q2, out2)
        l1.backward()
        l2.backward()
        opt1.step()
        opt2.step()
        p1, p2 = P1.clone(), P2.clone()
        p1.requires_grad, p2.requires_grad = True, True
        
    if i % printEpoch == 0:
        print('Epoch {} of {}\t\tElapsed time: {:.6} s'.format(i,epoch, time.time()-start))
    if i  % savingEpoch == 0:
        torch.save(n1.state_dict(), PATH1)
        torch.save(n2.state_dict(), PATH2)
        print('Saved weight files in {}'.format(FOLDER))


        

        
#AFTER THE GAME
#save params
torch.save(n1.state_dict(), PATH1)
torch.save(n2.state_dict(), PATH2)
print('Finished learning! Saved weight files in {}'.format(FOLDER))

#calculate total training time
elapsed = time.time() - start
print('\n\nTotal training time: {:.6} s'.format(elapsed))

#plot the reward
r1 = np.array(rew1)
r2 = np.array(rew2)
plt.plot(r1,'-b')
plt.plot(r2,'-r')
plt.show()
