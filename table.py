#class to manage the table: it handles business like counting the
#points, deciding who's got to deal and play first etc.

#players need to be labeled 0,1,2,3

import deck
import player
import random as rnd

#I am using tuples for sequences that I know won't be changed, like playerID or
#players, and list for everything else.

class Table:

    def __init__(self, Round, rules):     #rules is either 'Amsterdam' or 'Rotterdam'. So far only Rotterdam rules implemented
        self.nRound   = Round             #a game is made of 16 rounds
        self.playerID = 0,1,2,3      #the tuple's index is the player's name - 1
        self.cycleID = self.playerID * 2    #useful to cycle from player 4 to 1
        self.players  =[player.Player(i) for i in self.playerID]
        self.dealer   = rnd.choice(self.playerID)  #first dealer chosen randomly
        self.rules = rules
        self.Order()


    def Order(self):
        self.orderedPlayers = [self.players[self.cycleID[self.dealer + i + 1]] for i in range(len(self.playerID))]
        
    def WhoDeals(self):
        if self.dealer == len(self.playerID) - 1:
            self.dealer = 0
            self.Order()
            return self.dealer, self.players[self.dealer]
        else:
            self.dealer += 1
            self.Order()
            return self.dealer, self.players[self.dealer]

    def WhoPlays(self):
        p0   = self.orderedPlayers[0]
        p0ID = self.players.index(p0) 
        return p0ID, p0

    def DealCards(self, d):   #d is the deck instance object
        [p.hand.clear() for p in self.players]
        for p in self.orderedPlayers:
            p.hand = d.HandOutCards(self.orderedPlayers.index(p))
        
    def PlayCards(self):
        #the rules are going to be implemented in Player.play()
        #this function is going to stay more or less the same
        self.playedCards = [p.Play(self) for p in self.orderedPlayers]
        self.playedTuples = [c.CardAsTuple() for c in self.playedCards] #to check if the algorithm works on the command line
    
    def WhoWinsTrick(self):
        #this will have to be changed according to the rules
        winValue = self.playedCards[0].value     #the value of the winning card
        winner   = self.playedCards[0]        #the winning card        
        for c in self.playedCards:
            if c.value > winValue:
                winValue = c.value
                winner = c
        #TODO:add the points to the team's score
        return winner

    def whoWinsRound(self):
        #a function to be implemented after the rules are set
        #it determines what team has the highest points and returns it
        #it also increments their total (game) points
        pass


