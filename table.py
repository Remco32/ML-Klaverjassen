#class to manage the table: it handles business like counting the
#points, deciding who's got to deal and play first etc.

#players need to be labeled 0,1,2,3

import deck
import player
import random as rnd

#I am using tuples for sequences that I know won't be changed, like playerID or
#players, and list for everything else.

class Table:

    def __init__(self, r):
        self.nRound   = r              #a game is made of 16 rounds
        self.playerID = 0,1,2,3      #the tuple's index is the player's name - 1
        self.cycleID = self.playerID * 2    #useful to cycle from player 4 to 1
        self.players  =[player.Player(i) for i in self.playerID]
        self.dealer   = rnd.choice(self.playerID)  #first dealer chosen randomly


    def whoDeals(self):
        if self.dealer == len(self.playerID) - 1:
            self.dealer = 0
            return self.dealer
        else:
            self.dealer += 1
            return self.dealer
        
    def whoPlays(self):
        if self.dealer == len(self.playerID) - 1:  #if the dealer is the last player
            return 0
        else:
            return self.playerID[self.dealer + 1]#sinceindex = player's name

    def dealCards(self):
        [p.hand.clear() for p in self.players]
        for i in range(4):
            self.players[self.cycleID[self.dealer + i + 1]].hand = deck.handOutCards(i)
        self.whoDeals()   #update dealer
            
    def whoWinsTable(self):
        #a function to be implemented after the rules are set
        #it determines what player and what team win the 4 cards
        #on the table, and also increments their partial (round) points
        pass

    def whoWinsRound(self):
        #a function to be implemented after the rules are st
        #it determines what team has the highest points and returns it
        #it also increments their total (game) points
        pass


