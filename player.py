
#Class that represents players
#
#I am implementing simple rules to begin with; they are as simple as
# - if you're starting the trick, play any card; otherwise:
# - play in suit if possible
# - if not possible play a trump if possible
# - if not possible play any card
#
#I suggest we implement Rotterdam rules after because they seem easier according
#to this link https://www.pagat.com/jass/klaverjassen.html
#We should then implement Amsterdam rules as well, switching between
#the rules with the table.Table.rules variable
#
#The lines marked by ### are the lines where machine learning algoriths
#will take place
#

import random as rnd
import deck
import table
import learn
import torch
import torch.nn as nn
import numpy as np
import exploration_strategies as expl

class Player:

    # player 0 and 2 are in team 0; player 1 and 3 are in team 1 
    def __init__(self, number, alpha, y):  #number is an integer from 0 to 3
        self.position = number
        self.hand = []
        if number % 2 == 0:
            self.team = 0
        else:
            self.team = 1
        self.handSum = 0
            
        self.alpha  = alpha    #learning rate
        self.y      = y        #discount rate
        self.net    = learn.Net(70)     #the last 3 features are going to be the cards on the table
        self.opt    = torch.optim.SGD(self.net.parameters(), lr=self.alpha)
        self.loss   = nn.MSELoss()
        self.reward = 0
        self.rewardArray = []
        self.weightedRewardArray = []
        self.epsilon = 0.3 # exploration rate
        self.testing = False # Boolean to toggle whether the network is training (with exploration) or testing.

    """    
    def Pop(self):    #these things will be replaced in NetPlay(self, *args). Check whether all that's needed in here is also in NetPlay
        popped = self.subHand.pop(rnd.randrange(0,len(self.subHand)))
        self.hand.remove(popped)
        return popped
    """
        
    # method to play a card from the hand.
    def Play(self, tab, d):  #tab is the table object for the rules, d is the deck object for the trump suit
        
        if tab.WhoPlays()[0] == self:    #if he's starting the trick
            for c in self.hand:
                c.isPlayable = True
            self.subHand = [c for c in self.hand if c.isPlayable == True]
            self.played = self.NetPlay(tab, d)[0]
            self.hand.remove(self.played)
            for c in self.hand:
                c.isPlayable = False
                        
        else:
            if tab.rules == 'Simple':
                self.SimplePlay(tab, d)
            elif tab.rules == 'Rotterdam':
                self.RotterdamPlay(tab, d)
            elif tab.rules == 'Amsterdam':
                self.AmsterdamPlay(tab, d)
            else:
                print('Unknown rules. Input either \'Simple\', \'Amsterdam\' or \'Rotterdam\'')
        return self.played             #self.played is assigned within each of the play methods below

        
    def NetPlay(self, tbl, dck):
        #function to play a card using reinforcement learning
        #self.feat = self.net.CreatePlayFeaturesVector(self, tbl, dck)   #called in table.DealCards
        self.idPlayable = []
        for i,c in enumerate(self.feat):
            if i < 32:   #only the hand
                if c.item() == 1:
                    for card in self.subHand:
                        if card.index == i:
                            self.idPlayable.append(i)
        self.output = self.net(self.feat)

        idP = self.FindAllowedMaximum()     #BIG CHANGE: NO BACKPROP FOR ILLEGAL MOVES
        #Only changes the value for idP if an exploration step is taken, else uses idP given as the first argument
        if self.testing == False:
            idP = expl.diminishingEpsilonGreedy(idP, self.epsilon, self.idPlayable, tbl.currentEpoch, tbl.maximumEpoch)
        for c in self.subHand:
            if c.index == idP:
                cc = c
        return cc, idP


        #the rest of the training part (when the card has been played) needs to be done in table.py
        #in the method table.PlayCards() where each player plays a card and rewards are calculated



    def FindAllowedMaximum(self):
        with torch.no_grad():
            outFeat = self.output.clone().detach().numpy().tolist() #create a list
        outFeatSorted = sorted(outFeat, reverse=True)   #sort it in descending order
        element   = outFeatSorted[0]
        elementID = outFeat.index(element)
        ind = 0
        while elementID not in self.idPlayable:
            ind += 1
            element   = outFeatSorted[ind]
            elementID = outFeat.index(element)
        return elementID
            
        
    def SimplePlay(self, tab, d):
        playableCards = 0
        
        for c in self.hand:                     #first try to flag cards in suit as playable
            if c.suit == tab.leadingSuit:
                c.isPlayable = True
                playableCards += 1

        if playableCards == 0:                  #if none, then try flag trumps as playable (of course checking if playableCards == 0
            for c in self.hand:                 #also implies that trump != leading suit, so that cards are not counted twice)
                if c.suit == d.trumpSuit:
                    c.isPlayable = True
                    playableCards += 1

        if playableCards == 0:                  #otherwise flag any other card
            for c in self.hand:
                c.isPlayable = True 

        self.subHand = [c for c in self.hand if c.isPlayable == True]
        tmp = self.NetPlay(tab,d)
        self.played = tmp[0]     
        self.playedID =tmp[1]

        self.hand.remove(self.played)
        
        if self.hand != []:                       #after playing the card, all the others are flagged as unplayable before the next trick
            for c in self.hand:
                c.isPlayable = False

                
    def RotterdamPlay(self, tab, d):
        pass

    def AmsterdamPlay(self, tab, d):
        pass

