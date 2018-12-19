import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pdb
import os # For path creation for saving files

#VARIABLES
alpha, y, epoch, savingEpoch, printEpoch = 0.01, 0.9, 300, 300, 30 # Hyperparameters
rew1, rew2 = [], []
r1Win, r1Lose, r2Win, r2Lose = 3, -0.25, 2, -0.5

# Print time each epoch?
printElapsedTimeEachEpoch = True

#load parameters?
loadP = 1
#to save the weights
#FOLDER = '/Users/tommi/github/ML-Klaverjassen/simpler/weights/'
FOLDER = os.path.dirname(__file__) + '/weights/' # Using relative path

# Check if folder exists, to avoid errors
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)


PATH1 = FOLDER + 'net1_weights.pth'
PATH2 = FOLDER + 'net2_weights.pth'




#THE NETWORK
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conn1 = nn.Linear(12,20)
        self.conn2 = nn.Linear(20,6)
    def forward(self, x):
        x = torch.sigmoid(self.conn1(x))
        x = self.conn2(x)
        return x

#nets for the players   
n1 = Net()
n2 = Net()

#load parameters from previous training rounds
if loadP == 1 and os.path.exists(PATH1) and os.path.exists(PATH2):

    n1.load_state_dict(torch.load(PATH1))
    n2.load_state_dict(torch.load(PATH2))
    print('Loaded model parameters from folder {}'.format(FOLDER))

#optimisers and losses
opt1  = torch.optim.SGD(n1.parameters(), lr = alpha)
opt2  = torch.optim.SGD(n2.parameters(), lr = alpha)
loss1 = nn.MSELoss()
loss2 = nn.MSELoss()






#THE GAME
start_first_epochs = time.time()
for i in range(epoch):
    start_current_epoch = time.time()

    id1, id2 = [0, 2, 4], [1, 3, 5]
    p1 = torch.zeros(12, dtype=torch.float, requires_grad=True)
    p2 = torch.zeros(12, dtype=torch.float, requires_grad=True)
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

        with torch.no_grad():
            P1 = p1.clone()
            P2 = p2.clone()
            P1[6 + a1] = 1
            P2[6 + a1] = 1
            p1 = P1.clone()
            p2 = P1.clone()
            p1.requires_grad = True
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

        if a1 > a2:
            r1 = r1Win
            r2 = r2Lose
            rew1.append(r1)
            rew2.append(r2)
        if a2 > a1:
            r2 = r2Win
            r1 = r1Lose
            rew1.append(r1)
            rew2.append(r2)

        with torch.no_grad():
            P1 = p1.clone()
            P2 = p2.clone()
            P1[a1] = 0
            P2[a2] = 0
            P1[6 + a2] = 2       #major change
            P2[6 + a2] = 2       #major change

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
        l2 = loss2(Q1, out2)
        l1.backward()
        l2.backward()
        opt1.step()
        opt2.step()
        p1, p2 = P1.clone(), P2.clone()
        p1.requires_grad, p2.requires_grad = True, True
        
    if i % printEpoch == 0:
        print('Epoch {} of {}'.format(i,epoch))
    if i  % savingEpoch == 0:
        torch.save(n1.state_dict(), PATH1)
        torch.save(n2.state_dict(), PATH2)
        print('Epoch {} of {}\nSaved weight files in {}'.format(i,epoch,FOLDER))

    # Show time spent
    if printElapsedTimeEachEpoch:
        print('Elapsed time this epoch: ' + str(time.time() - start_current_epoch) + ' seconds')

#AFTER THE GAME
#save params
torch.save(n1.state_dict(), PATH1)
torch.save(n2.state_dict(), PATH2)
print('Epoch {} of {}\nSaved weight files in {}'.format(i,epoch,FOLDER))

#calculate total training time
elapsed = time.time() - start_first_epochs
print('\n\nTotal training time: {:.6}'.format(elapsed))

#plot the reward
r1 = np.array(rew1)
r2 = np.array(rew2)
plt.plot(r1,'-b')
plt.plot(r2,'.r')
plt.show()
