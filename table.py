#class to manage the table: it handles business like counting the
#points, deciding who's got to deal and play first etc.
#
#players need to be labeled 0,1,2,3
#
#

import deck
import player
import random as rnd


class Table:

    def __init__(self, Round, rules):       #rules is either 'Simple, 'Amsterdam' or 'Rotterdam'. So far only 'Simple' rules implemented
        self.nRound   = Round               #a game is made of 16 rounds
        self.playerID = 0,1,2,3             #the tuple's index is the player's name 
        self.cycleID = self.playerID * 2    #useful to cycle from player 4 to 1
        self.players  =[player.Player(i) for i in self.playerID]
        self.dealer   = rnd.choice(self.playerID)  #first dealer chosen randomly
        self.rules = rules
        self.roundScore = [0, 0]
        self.gameScore  = [0, 0]
        self.Order(self.dealer + 1)           #ordering the players with respect to the PLAYER STARTING THE TRICK (refer to cycleID, this
                                              #means 3+1=0)
       

                                                
    def Order(self, who):
        self.orderedPlayers = [self.players[self.cycleID[who + i]] for i in range(len(self.playerID))]

        
    def WhoDeals(self):
        if self.dealer == len(self.playerID) - 1:
            self.dealer = 0
            self.Order(self.dealer + 1)
            return self.dealer, self.players[self.dealer]
        else:
            self.dealer += 1
            self.Order(self.dealer + 1)
            return self.dealer, self.players[self.dealer]

        
    def WhoPlays(self):
        p0   = self.orderedPlayers[0]
        p0ID = self.players.index(p0) 
        return p0ID, p0

    
    def DealCards(self, d):   #d is the deck object
        [p.hand.clear() for p in self.players]
        for p in self.orderedPlayers:
            p.hand = d.HandOutCards(self.orderedPlayers.index(p))
        

    def PlayCards(self, d):     #d is the deck object
        for p in self.orderedPlayers:
            if self.orderedPlayers.index(p) == 0:
                self.playedCards = [p.Play(self, d)]   #play the first card
                self.leadingSuit = self.playedCards[0].suit
            else:
                self.playedCards.append(p.Play(self, d))
        self.playedTuples = [c.CardAsTuple() for c in self.playedCards] #to check from the command line if the algorithm works 

        
    def WhoWinsTrick(self, d):    #d is the deck object
        trickPoints = 0
        self.winValue = 0
        self.playedTrump = False
        for c in self.playedCards:
            trickPoints += c.value
            if c.suit == d.trumpSuit:
                self.playedTrump = True

        if self.playedTrump == True:
            for c in self.playedCards:
                if c.suit == d.trumpSuit:
                    if c.value >= self.winValue:
                        self.winValue = c.value
                        self.winnerCard = c
        else:
            for c in self.playedCards:
                if c.suit == self.leadingSuit:
                    if c.value >= self.winValue:
                        self.winValue = c.value
                        self.winnerCard = c
            
        tmp = self.playedCards.index(self.winnerCard)
        self.winnerPlayer = self.orderedPlayers[tmp]
        self.winnerPlayerID = self.players.index(self.winnerPlayer)
        self.roundScore[self.winnerPlayer.team] += trickPoints    #trick points assignment
        self.Order(self.winnerPlayerID)                           #the game starts from the trick winner
        
        return self.winnerPlayer

    
    def whoWinsRound(self):
        pass


