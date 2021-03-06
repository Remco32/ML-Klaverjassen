# Class to represent the deck in the game



import random as rnd
from collections import OrderedDict


class Card:
    def __init__(self, name, suit, value, rank):
        self.suit        = suit
        self.name        = name
        self.value       = value
        self.rank        = rank
        self.isPlayable  = False        
        self.whoPlayedMe = 0
    def SetValue(self, value):
        self.value = value
    def SetRank(self, rank):
        self.rank = rank
    def SetIndex(self, index):
        self.index = index
    def CardAsTuple(self):         
        return self.index, self.name, self.suit, self.value

class Deck:
    # assign values to the cards
    suits = ('d', 'c', 'h', 's')
    nonTrumpDict = OrderedDict([('A',(11,8)), ('10',(10,7)), ('K',(4,6)), ('Q',(3,5)), ('J',(2,4)), ('9',(0,3)), ('8',(0,2)), ('7',(0,1))])
    trumpDict    = OrderedDict([('J',(20,16)), ('9',(14,15)), ('A',(11,14)), ('10', (10,13)), ('K',(4,12)), ('Q',(3,11)), ('8',(0,10)), ('7',(0,9))])
    def __init__(self):   #create a trumpless deck with Klaverjassen cards
        self.cards = [Card(n, s,  self.nonTrumpDict[n][0], self.nonTrumpDict[n][1]) for s in self.suits for n in self.nonTrumpDict.keys()]
        [self.cards[i].SetIndex(i) for i in range(len(self.cards))] 
        self.dividedCards = self.handP0, self.handP1, self.handP2, self.handP3 = [], [], [], []
        self.maxCardValue = 20.   
    def SetTrump(self, trumpSuit):  #set the trump values
        self.cards = []
        self.__init__()
        self.trumpSuit = trumpSuit
        [c.SetValue(self.trumpDict[n][0]) for c in self.cards for n in self.trumpDict.keys() if c.name == n if c.suit == self.trumpSuit]
        [c.SetRank(self.trumpDict[n][1]) for c in self.cards for n in self.trumpDict.keys() if c.name == n if c.suit == self.trumpSuit]
    def DivideCards(self):     # Split the shuffled deck in four parts, so each player can receive their hand
        self.shuffledCards = rnd.sample(self.cards, len(self.cards))     
        shuffledDeck = [c for c in self.shuffledCards]  
        [self.dividedCards[i].clear() for i in range(4)]
        for c in shuffledDeck:
            c.whoPlayedMe = 0
        for cardIndex in range(len(shuffledDeck)):
            if cardIndex >= 0 and cardIndex <= 7:
                self.handP0.append(shuffledDeck[cardIndex])
            if cardIndex >= 8 and cardIndex <= 15:
                self.handP1.append(shuffledDeck[cardIndex])
            if cardIndex >= 16 and cardIndex <= 23:
                self.handP2.append(shuffledDeck[cardIndex])
            if cardIndex >= 24 and cardIndex <= 32:
                self.handP3.append(shuffledDeck[cardIndex])
    def HandOutCards(self, playerPosition):        # Hand out the cards to players
        return self.dividedCards[playerPosition]
