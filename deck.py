# Complete deck

import random as rnd


# class to represent a card in the game
class Card:

    def __init__(self, name, suit, value):
        self.suit       = suit
        self.name       = name
        self.value      = value
        self.isPlayable = False         #major change for playing algorithms

    def SetValue(self, value):
        self.value = value

    def SetIndex(self, index):
        self.index = index

    def CardAsTuple(self):          # useful to visualise the card; all the internal working are done with the Card object directly
        return self.index, self.name, self.suit, self.value

# the deck is now a class
class Deck:

    #suit tuple  (not "self" because every deck shares the same suits)
    suits = ('d', 'c', 'h', 's')
    #NEW value dictionaries to build a deck after deciding the trump in each round
    nonTrumpDict = {'A':11, '10':10, 'K':4, 'Q':3, 'J':2, '9':0, '8':0, '7':0}
    trumpDict    = {'J':20, '9':14, 'A':11, '10':10, 'K':4, 'Q':3, '8':0, '7':0}

    def __init__(self):   #create a trumpless deck with Klaverjassen cards
        self.cards = [Card(n, s,  self.nonTrumpDict[n]) for s in self.suits for n in self.nonTrumpDict.keys()]
        [self.cards[i].SetIndex(i) for i in range(len(self.cards))] #now cards are uniquely indexed for the whole game, after shuffling too #TODO nope, the order of the indexed cards isn't set, seems random
        self.dividedCards = self.handP0, self.handP1, self.handP2, self.handP3 = [], [], [], []

    def SetTrump(self, trumpSuit):  #set the trump values
        self.cards = []
        self.__init__()
        self.trumpSuit = trumpSuit
        [c.SetValue(self.trumpDict[n]) for c in self.cards for n in self.trumpDict.keys() if c.name == n if c.suit == self.trumpSuit]
        
    # Split the shuffled deck in four parts, so each player can receive their hand
    def DivideCards(self):
        #the hands need to be emptied beforehand, otherwise cards will be
        #added indefinitely; then the cards need to be shuffled again
        self.shuffledCards = rnd.sample(self.cards, len(self.cards))      #now a copy of the deck is shuffled
        shuffledDeck = [c for c in self.shuffledCards] 
        [self.dividedCards[i].clear() for i in range(4)]

        for cardIndex in range(len(shuffledDeck)):
            if cardIndex >= 0 and cardIndex <= 7:
                self.handP0.append(shuffledDeck[cardIndex])
            if cardIndex >= 8 and cardIndex <= 15:
                self.handP1.append(shuffledDeck[cardIndex])
            if cardIndex >= 16 and cardIndex <= 23:
                self.handP2.append(shuffledDeck[cardIndex])
            if cardIndex >= 24 and cardIndex <= 32:
                self.handP3.append(shuffledDeck[cardIndex])

    # A player can request his hand from the deck using this function
    def HandOutCards(self, playerPosition):    
        return self.dividedCards[playerPosition]
