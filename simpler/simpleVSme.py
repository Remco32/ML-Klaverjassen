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




#THE GAME
start = time.time()
rounds = input('How many rounds?  --->')
for j in range(int(rounds)):
    
    idP = [1, 2, 5]       #for now, since I only trained the algorithm with these
    cards = [0, 3, 4]     #cards, he has no adaptability. Useful to train with other
                          #combinations as well
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
        pl = n(p)
        played = pl.argmax().item()
        print(played)     #all the updates to the hand are to be done after the
                          #card has been payed (as in simple.py) and it makes sense
                          #given the interpretation of the features
        p[played]     = 0
        p[7 + played] = 1
        p[7 + int(c)] = 2
        idP.remove(played)
        cards.remove(int(c))
 


        

        
