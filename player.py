#Class that represents players

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
        self.alpha  = alpha    
        self.y      = y       
        self.net    = learn.Net(97)     
        self.opt    = torch.optim.SGD(self.net.parameters(), lr=self.alpha)
        self.loss   = nn.MSELoss()
        self.reward = 0
        self.rewardArray = []
        self.weightedRewardArray = []
        self.epsilon = 0.3 
        self.testing = False
        
    def Pop(self):    
        popped = self.subHand.pop(rnd.randrange(0,len(self.subHand)))
        return popped

    def handAsTuple(self):
        h = [c.CardAsTuple() for c in self.hand]
        return h

    #general play routine
    def Play(self, tab, d):
        self.feat = self.net.UpdateFeatureVectors(self, tab, d)[0]
        if tab.WhoPlays()[0] == self:    
            for c in self.hand:
                c.isPlayable = True
            if self.behaviour == 'Network':
                self.subHand = [c for c in self.hand if c.isPlayable == True]
                self.played = self.NetworkPlay(tab, d)[0] 
                if len(self.subHand) == 1:
                    self.played = self.subHand[0]            
                for c in self.hand:
                    c.isPlayable = False
            elif self.behaviour == 'Random':
                self.subHand = [c for c in self.hand if c.isPlayable == True]
                self.played = self.RandomPlay(tab, d)
                for c in self.hand:
                    c.isPlayable = False
            elif self.behaviour in ('MonteCarlo', 'MonteCarlo2'):
                self.subHand = [c for c in self.hand if c.isPlayable == True]
                self.played = self.MonteCarloPlay(tab, d)
                for c in self.hand:
                    c.isPlayable = False
        else:
            if self.behaviour == 'Random':
                self.played = self.RandomPlay(tab, d)
            elif self.behaviour == 'Network':
                self.played = self.NetworkPlay(tab, d)[0]
                if len(self.subHand) == 1:
                    self.played = self.subHand[0]
            elif self.behaviour in ('MonteCarlo', 'MonteCarlo2'):
                self.played = self.MonteCarloPlay(tab, d)
            else:
                print('Unknown rules. Input either \'Simple\', \'Amsterdam\'')
        if self.behaviour != 'MonteCarlo':
            self.hand.remove(self.played)
        return self.played

    #To play a card using the neural network
    def NetPlay(self, tbl):
        global cc
        self.idPlayable = []
        for i, c in enumerate(self.feat):
            if i < 32: 
                if c.item() == 1:
                    for card in self.subHand:
                        if card.index == i:
                            self.idPlayable.append(i)
        idP = self.FindAllowedMaximum(self.feat) 
        if self.testing == False:
            idP = expl.diminishingEpsilonGreedy(idP, self.epsilon, self.idPlayable, tbl.currentEpoch, tbl.maximumEpoch)
        for c in self.subHand:
            if c.index == idP:
                cc = c
        return cc, idP

    def FindAllowedMaximum(self, feat):   
        self.output = self.net(feat)
        with torch.no_grad():
            outFeat = self.output.clone().detach().numpy().tolist() 
        outFeatSorted = sorted(outFeat, reverse=True) 
        element   = outFeatSorted[0]
        elementID = outFeat.index(element)
        for element in outFeatSorted:
            elementID = outFeat.index(element)
            if elementID in self.idPlayable:
                break
        return elementID           
        
    def RandomPlay(self, tab, d):    
        playableCards = 0
        for c in self.hand:  
            if c.suit == tab.leadingSuit:
                c.isPlayable = True
                playableCards += 1
        if playableCards == 0:                 
            for c in self.hand:                 
                if c.suit == d.trumpSuit:
                    c.isPlayable = True
                    playableCards += 1
        if playableCards == 0:                 
            for c in self.hand:
                c.isPlayable = True
        self.subHand = [c for c in self.hand if c.isPlayable == True]
        tmp = self.Pop()
        if self.hand != []:                       
            for c in self.hand:
                c.isPlayable = False
        return tmp

    def NetworkPlay(self, tab, d):  
        playableCards = 0
        for c in self.hand:  
            if c.suit == tab.leadingSuit:
                c.isPlayable = True
                playableCards += 1
        if playableCards == 0:
            for c in self.hand: 
                if c.suit == d.trumpSuit:
                    c.isPlayable = True
                    playableCards += 1
        if playableCards == 0: 
            for c in self.hand:
                c.isPlayable = True
        self.subHand = [c for c in self.hand if c.isPlayable == True]
        tmp = self.NetPlay(tab)
        if self.hand != []: 
            for c in self.hand:
                c.isPlayable = False
        return tmp

    def MonteCarloPlay(self, tab, d):     
        if self.behaviour == "MonteCarlo2":
            tmp = self.played
        else:
            playableCards = 0
            for c in self.hand: 
                if c.suit == tab.leadingSuit:
                    c.isPlayable = True
                    playableCards += 1
            if playableCards == 0:
                for c in self.hand:  
                    if c.suit == d.trumpSuit:
                        c.isPlayable = True
                        playableCards += 1
            if playableCards == 0:  
                for c in self.hand:
                    c.isPlayable = True
            self.subHand = [c for c in self.hand if c.isPlayable == True]
            tmp = self.Pop()
        if self.hand != []:  
            for c in self.hand:
                c.isPlayable = False
        return tmp
