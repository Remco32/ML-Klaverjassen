"""
To see if the AI after training with simple.py can play and win againt a
human player.
So no training, only using the trained data
"""


import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pdb



alpha, y = 0.1, 0.9

#load parameters?
loadP = 1
#to save the weights
FOLDER = '/Users/tommi/github/ML-Klaverjassen/simpler/weights/'
PATH = FOLDER + 'net2_weights.pth'






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
n = Net()

#load parameters from previous training rounds
if loadP == 1:
    n.load_state_dict(torch.load(PATH))
    print('Loaded model parameters from {}'.format(PATH))

opt  = torch.optim.SGD(n.parameters(), lr = alpha)
loss = nn.MSELoss()



#THE GAME
start = time.time()
rounds = input('How many rounds?  --->')
for j in range(int(rounds)):
    
    idP = [3, 4, 5]       
    cards = [0, 1, 2]     
    p = torch.zeros(13, dtype=torch.float, requires_grad=True)
    for k in range(13):
            if k in idP:
                p[k] = 1

    print('Round {}'.format(j))
    for i in range(3):
        c = input('What card to play among {}?  --->'.format(cards))
        while int(c) not in cards:
            c = input('Invalid choice. Choose among {}  --->'.format(cards))
            
        p[6] = int(c)
        opt.zero_grad()
        pl = n(p)
        played = pl.argmax().item() #TODO: add backpropagation when an illegal card is selected; not working so far

        while played not in idP:
            r = -1
            with torch.no_grad():
                Q = pl.clone()
                Q[played] += alpha * ( r + y * torch.max(pl).item() - Q[played])
            l = loss(Q, pl)
            l.backward()
            opt.step()
            opt.zero_grad()
            pl = n(p)
            played   = pl.argmax().item()
        print(played)     #all the updates to the hand are to be done after the
                          #card has been played (as in simple.py) and it makes sense
                          #given the interpretation of the features
        with torch.no_grad():            
            p[played]     = 0
            p[7 + played] = 1
            p[7 + int(c)] = 2
        idP.remove(played)
        cards.remove(int(c))
 


        

        
