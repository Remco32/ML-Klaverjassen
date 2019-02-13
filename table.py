#class to manage the table: it handles business like counting the
#points, deciding who's got to deal and play first etc.
#
#players need to be labeled 0,1,2,3
#
#
import os

import deck
import player
import random as rnd
import learn
import torch
import numpy as np


class Table:

    def __init__(self, Round, alpha, y):    #now all players have same learning rate and y. For exp. purposes it might be interesting to have different values
        self.playerID = 0,1,2,3             #the tuple's index is the player's name 
        self.cycleID = self.playerID * 2    #useful to cycle from player 4 to 1
        self.players  =[player.Player(i, alpha, y) for i in self.playerID]
        self.dealer   = rnd.choice(self.playerID)  #first dealer chosen randomly
        self.roundScore = [0, 0]    # No longer cumulative
        self.gameScore  = [0, 0]
        self.playedCards = []
        self.Order(self.dealer + 1)           #ordering the players with respect to the PLAYER STARTING THE TRICK (refer to cycleID, this means 3+1=0)
        
        # For variables used in running the experiments
        self.currentEpoch = 0
        self.maximumEpoch = 0
        self.testingScores = []      #it will become a 2xN matrix, where N is the number of testing rounds
        self.testingCycleScoresTeam0 = [] #Fills up with the average score of a team for each cycle
        self.testingCycleScoresTeam1 = []
        self.testingTeam0WinsThisCycle = 0   # To store the wins during testing cycle
        self.testingTeam1WinsThisCycle = 0
        self.testingWinRatioTeam0 = []
        self.testingTeam0IncrementalWins = []
        self.testingTeam1IncrementalWins = []

        self.testingTeam0TotalWins = []  # To store the total wins during testing
        self.testingTeam1TotalWins = []

        self.oldDiscountRate = y
       


    def SetPlayerBehaviour(self, playerID, behaviour):   #behaviour is either 'Random' or 'Network'
        self.players[playerID].behaviour = behaviour

        
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
        return p0, p0ID



    def LoadState(self, path):
        for player in self.players:
            player.net.load_state_dict(torch.load(path))
        print("Loaded model parameters")


    def SaveState(self, path):
        for player in self.players:

            filePath = path + 'player' + str(player.position) + '_weights.pth'
            torch.save(player.net.state_dict(), filePath)

            
    def DealCards(self, d):   #d is the deck object
        self.allPlayedCards = {}         #to keep track (count) of all the played cards and who played them
        self.roundScore = [0,0] # Reset to 0, instead of cummulating

        [p.hand.clear() for p in self.players]        
        for p in self.orderedPlayers:
            p.handSum = 0.
            p.hand = d.HandOutCards(self.orderedPlayers.index(p))
            p.feat = p.net.CreatePlayFeaturesVector(p, self, d)    #create the feature vector when the cards are dealt
            for card in p.hand:
                p.handSum += card.value
               

    def PlayCards(self, d):     #d is the deck object
        #playedCards and cardsOnTable are really the same thing except
        #cardsOnTable has one less element (it doesn't have the last played card
        #because no player cares about that)
        self.cardsOnTable = []
        self.leadingSuit = 'none'     #changed when the first player plays
        for i,p in enumerate(self.orderedPlayers):
            if i == 0:
                self.playedCards  = [p.Play(self, d)]   #play the first card
                self.leadingSuit  = self.playedCards[0].suit
            else:
                self.playedCards.append(p.Play(self, d))
                
            self.playedCards[i].whoPlayedMe = p.position 
            
        for c in self.playedCards:             
            tmp = self.playedCards.index(c)
            tmp = self.orderedPlayers[tmp]    #player object who played the card
            tmp = self.players.index(tmp)     #player number who played the card
            self.allPlayedCards[c] = tmp      #add to the dict
            
        self.playedTuples = [c.CardAsTuple() for c in self.playedCards] #to check from the command line if the algorithm works 


    """
    For now we couldn't think of a way to implement Q-learning after each round is finished, since there is no next-state to go to
    after the round is over. While we think about that, we implement Q-learning after each trick with reward +1 to the winning team
    and reward -1 to the losing team.
    """    
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
        for p in self.players:
            if p.team == self.winnerPlayer.team:
                p.reward = 1.0 + 8 - len(p.hand)
                #if p == self.winnerPlayer:
                #    p.reward += 0.2      #reward good plays to single player
            else:
                p.reward = -1.0 - (8 - len(p.hand))
            #p.rewardArray.append(p.reward)

        self.Order(self.winnerPlayerID)                           #the game starts from the trick winner


        """
        # Increment wincount if we are testing
        if self.winnerPlayer.testing:
           if self.winnerPlayer.team == 0:
               self.testingTeam0WinsThisCycle += 1
               self.testingTeam0IncrementalWins.append(self.testingTeam0WinsThisCycle)
               self.testingTeam1IncrementalWins.append(self.testingTeam1WinsThisCycle)

           else:
               self.testingTeam1WinsThisCycle += 1
               self.testingTeam1IncrementalWins.append(self.testingTeam1WinsThisCycle)
               self.testingTeam0IncrementalWins.append(self.testingTeam0WinsThisCycle)

        """
        """
        # If dealing player has one card: set discount rate to 0 and apply reward
        if len(self.players[self.dealer].hand) == 1:
            for p in self.players:
                # Set discount rate to 0
                #p.y = 0
                if p.team == self.winnerPlayer.team:
                    p.reward = 10
                else:
                    p.reward = -10
        """


        # End of round
        if self.checkEmptyHands():
            self.roundScore[self.winnerPlayer.team] += 10

            if self.winnerPlayer.testing:
                if self.winnerPlayer.team == 0:
                    self.testingTeam0WinsThisCycle += 1
                    self.testingTeam0IncrementalWins.append(self.testingTeam0WinsThisCycle)
                    self.testingTeam1IncrementalWins.append(self.testingTeam1WinsThisCycle)

                else:
                    self.testingTeam1WinsThisCycle += 1
                    self.testingTeam1IncrementalWins.append(self.testingTeam1WinsThisCycle)
                    self.testingTeam0IncrementalWins.append(self.testingTeam0WinsThisCycle)
            """
            #Set discount rate back
            for p in self.players:
                p.y = self.oldDiscountRate
            
            for p in self.players:
                if p.team == self.winnerPlayer.team:
                    p.reward = 10
                else:
                    p.reward = -10
            """
            # For exploration
            if not self.players[0].testing:
                self.currentEpoch += 1
        
        return self.winnerPlayer

    #Check if all the players played their cards
    def checkEmptyHands(self):

        if (len(self.players[0].hand) == 0 and len(self.players[1].hand) == 0
                and len(self.players[2].hand) == 0 and len(self.players[3].hand) == 0):
            return True
        return False

    def DoBackprop(self):
        with torch.no_grad():

            Plist = [f.feat.clone() for f in self.orderedPlayers]
            P = [Pl.clone() for Pl in Plist]  # clone the feature vectors to delete played card for next move
                
            for i,feat_vec in enumerate(P):     #for every player's features
                for j,c in enumerate(feat_vec):         #for every card in the feature vector
                    if j == self.playedCards[i].index:          #if the card's index is the index of the card played by that player
                        feat_vec[j] = 0                #set the value to 0
            q = []
            Q = [p.output.clone() for p in self.orderedPlayers]    #clone of the output
        for i,p in enumerate(self.orderedPlayers):                #check simple3.py for reference
            j = self.playedCards[i].index
            with torch.no_grad():
                q.append(p.net(P[i]))                              #output from the new state                        
                Q[i][j] += p.alpha * (p.reward + p.y * torch.max(q[i]).item() - Q[i][j])       #Q-learning formula 
            p.l = p.loss(p.output, Q[i]) #compute loss
            p.opt.zero_grad()
            p.l.backward()                #do backprop
            p.opt.step()                  #adjust weights after backprop
            #p.opt.zero_grad()
            ''' This is probably not necessary since the feature vectors are updated in players.Play(), 
                being built from scratch with the updated information.
            p.feat = P[i].clone()            #update the feature vector
            p.feat.requires_grad = True
            '''

    
    def whoWinsRound(self):
        pass

    def calculateTestResults(self):
        # Calculate scores

        # Convert to an array
        array = np.array(self.testingScores, dtype='int')

        #Calculate scores for end of rounds
        scoreTeam0 = 0
        scoreTeam1 = 0
        j = 0

        for i in range(len(self.testingScores)):

            if i % 8 == 7: # Only use the final score of a round
                scoreTeam1 += array[i,1]
                scoreTeam0 += array[i,0]
                j += 1

        self.testingCycleScoresTeam0.append(scoreTeam0/j)
        self.testingCycleScoresTeam1.append(scoreTeam1/j)

        #Calculate winrate

        winRatio = self.testingTeam0WinsThisCycle / (self.testingTeam0WinsThisCycle + self.testingTeam1WinsThisCycle)

        if not self.testingTeam0TotalWins or not self.testingTeam1TotalWins: # Empty lists
            self.testingTeam0TotalWins.append(self.testingTeam0WinsThisCycle)
            self.testingTeam1TotalWins.append(self.testingTeam1WinsThisCycle)
        else: # Append the cumulative wins
            self.testingTeam0TotalWins.append(self.testingTeam0WinsThisCycle + self.testingTeam0TotalWins[-1])
            self.testingTeam1TotalWins.append(self.testingTeam1WinsThisCycle + self.testingTeam1TotalWins[-1])

        self.testingTeam0WinsThisCycle = 0
        self.testingTeam1WinsThisCycle = 0

        self.testingWinRatioTeam0.append(winRatio)
