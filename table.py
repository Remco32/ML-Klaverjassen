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
        self.order()


    def order(self):
        self.orderedPlayers = [self.players[self.cycleID[self.dealer + i + 1]] for i in range(len(self.playerID))]
        
    def whoDeals(self):
        if self.dealer == len(self.playerID) - 1:
            self.dealer = 0
            self.order()
            return self.dealer, self.players[self.dealer]
        else:
            self.dealer += 1
            self.order()
            return self.dealer, self.players[self.dealer]

    def whoPlays(self):
        p0   = self.orderedPlayers[0]
        p0ID = self.players.index(p0) 
        return p0ID, p0

    def dealCards(self):
        [p.hand.clear() for p in self.players]
        for p in self.orderedPlayers:
            p.hand = deck.handOutCards(self.orderedPlayers.index(p))
        
    def playCards(self):
        #the rules are going to be implemented in Player.play()
        #this function is going to stay more or less the same
        self.playedCards = [p.play() for p in self.orderedPlayers]
        return self.playedCards
    
    def whoWinsTable(self):
        #this will have to be changed according to the rules
        winValue = self.playedCards[0][2]     #the value of the winning card
        winner   = self.playedCards[0]        #the winning card        
        for c in self.playedCards:
            if c[2] > winValue:
                winValue = c[2]
                winner = c
        #TODO:add the points to the team's score
        return winner

    def whoWinsRound(self):
        #a function to be implemented after the rules are set
        #it determines what team has the highest points and returns it
        #it also increments their total (game) points
        pass


