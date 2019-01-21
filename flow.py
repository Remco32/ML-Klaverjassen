import os

import table
import deck
import random as rnd
import learn
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

epochs, printEpoch, saveEpoch = 10000, 100, 10
# Load parameters?
#loadP = 0 #TODO check if loading works like it should
#Create folder if needed
#SAVEFOLDER = os.path.dirname(__file__) + '/weights/'

#if not os.path.exists(SAVEFOLDER):
#    os.makedirs(SAVEFOLDER)

#SAVEFILELIST = ['player' + str(i) + '_weights.pth' for i in range(4)]
#print(SAVEFILELIST)

"""
STEPS TO UNDERTAKE TO PROPERLY RUN THE PROGRAM:

######################
Initialise the table object: it automatically initialises the player objects, which in turn initialise the neural networks.
Initialise the deck, which in turn already shuffles.


######################
A round in the game: 

Set the trump suit.
Divide the cards.
Deal the cards to the players; this also creates the starting feature vectors for each player once they receive their hand.
OPT: Print the hands of the players to check the game from command line.
Until the players have cards in the hands, they play a (legal) card.
OPT: After all players have played their cards, a control printout is performed.
The trick winner is determined, rewards are assigned and then the table object performs the backprop for each player.

####################
Multiple rounds in the game to properly train the network. Same as above, repeated for a number of epochs, with some more control printouts.
"""


t = table.Table(16, 'Simple', 0.01, 0.9)
t.maximumEpoch = epochs # Set total epochs in table
d = deck.Deck()
#load parameters here
#if loadP == 0:
#    t.LoadState(SAVEFILELIST)
#elif loadP == 0:
#    print("Model parameters not loaded")

start = time.time()
    
for currentEpoch in range(epochs):

    #Set experiment information in tabnle
    t.currentEpoch = currentEpoch

    d.SetTrump(rnd.choice(d.suits))  
    d.DivideCards()
    t.DealCards(d)

    
    while t.players[0].hand != []:
        t.PlayCards(d)
        winner = t.WhoWinsTrick(d)
        t.DoBackprop()

    if currentEpoch % printEpoch == 0: print("Epoch {} of {} \t\t\tElapsed time: {:.4} s".format(currentEpoch, epochs, time.time() - start))
#    if currentEpoch == 100: 
#        t.SaveState(SAVEFOLDER)
#        print("Saved model parameters")

print("Training completed succesfully! \tElapsed time: {:.4} s \nShowing plots...".format(time.time() - start))

graphs  = [np.array(p.rewardArray) for p in t.players]
wGraphs = [np.array(p.weightedRewardArray) for p in t.players]
styles = ['-b', '-r', '-g', '-k']
plt.subplot(121)
p =[plt.plot(np.cumsum(graphs[i]), s, label='Player '+str(i)) for i,s in enumerate(styles)]
plt.legend()
plt.title('Absolute reward')
plt.xlabel('Tricks')
plt.ylabel('Reward')
plt.grid()
plt.subplot(122)
pp =[plt.plot(np.cumsum(wGraphs[i]), s, label='Player '+str(i)) for i,s in enumerate(styles)]
plt.legend()
plt.title('Weighted reward')
plt.xlabel('Tricks')
plt.ylabel('Reward / hand value')
plt.grid()
plt.show()

print("Program terminated! \t\tTotal running time: {:.5} s".format(time.time() - start))
