
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

    def __init__(self, number, alpha, y): 
        self.position = number
        self.hand = []
        self.behaviour = 'Network'
        if number % 2 == 0:
            self.team = 0
        else:
            self.team = 1
        self.handSum = 0
            
        self.alpha  = alpha    #learning rate
        self.y      = y        #discount rate
        self.net    = learn.Net(97)     
        self.opt    = torch.optim.SGD(self.net.parameters(), lr=self.alpha)
        self.loss   = nn.MSELoss()
        self.reward = 0
        self.rewardArray = []
        self.weightedRewardArray = []
        self.epsilon = 0.3 # exploration rate
        self.testing = False # Boolean to toggle whether the network is training (with exploration) or testing.


    
    def Pop(self):    
        popped = self.subHand.pop(rnd.randrange(0,len(self.subHand)))
        return popped
       

    def handAsTuple(self):
        h = [c.CardAsTuple() for c in self.hand]
        return h
    
    def Play(self, tab, d):
        self.feat = self.net.UpdateFeatureVectors(self, tab, d)[0]
        if tab.WhoPlays()[0] == self:    #if he's starting the trick
            for c in self.hand:
                c.isPlayable = True
            if self.behaviour == 'Network':
                self.subHand = [c for c in self.hand if c.isPlayable == True]
                self.played = self.NetworkPlay(tab, d)[0]
                for c in self.hand:
                    c.isPlayable = False
            elif self.behaviour == 'Random':
                self.subHand = [c for c in self.hand if c.isPlayable == True]
                self.played = self.RandomPlay(tab, d)
                for c in self.hand:
                    c.isPlayable = False
                        
        else:
            if self.behaviour == 'Random':
                self.played = self.RandomPlay(tab, d)
            elif self.behaviour == 'Network':
                self.played = self.NetworkPlay(tab, d)[0]
            else:
                print('Unknown rules. Input either \'Simple\', \'Amsterdam\'')

        self.hand.remove(self.played)
        return self.played             #self.played is assigned within each of the play methods below

        
    def NetPlay(self, tbl, dck):
        global cc #(Possibly) Fixes assignment error breaking the program after many iterations
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


    def FindAllowedMaximum(self):
        with torch.no_grad():
            outFeat = self.output.clone().detach().numpy().tolist() #create a list
        outFeatSorted = sorted(outFeat, reverse=True)  #sort it in descending order
        element   = outFeatSorted[0]
        elementID = outFeat.index(element)
        for element in outFeatSorted:
            elementID = outFeat.index(element)
            if elementID in self.idPlayable:
                break
        return elementID
            
        
    def RandomPlay(self, tab, d):     #play a random card in AMS rules
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
        tmp = self.Pop()
        if self.hand != []:                       #after playing the card, all the others are flagged as unplayable before the next trick
            for c in self.hand:
                c.isPlayable = False
        return tmp

    def NetworkPlay(self, tab, d):      #play with neural network in AMS rules
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
        #self.played = tmp[0]
        #self.playedID =tmp[1]
        if self.hand != []:                       #after playing the card, all the others are flagged as unplayable before the next trick
            for c in self.hand:
                c.isPlayable = False

        return tmp
