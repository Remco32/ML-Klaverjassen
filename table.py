import os
import deck
import player
import random as rnd
import learn
import torch
import numpy as np


class Table:

    def __init__(self, Round, alpha, y):    
        self.playerID = 0,1,2,3              
        self.cycleID = self.playerID * 2    
        self.players  =[player.Player(i, alpha, y) for i in self.playerID]
        self.dealer   = rnd.choice(self.playerID)   
        self.roundScore = [0, 0]     
        self.gameScore  = [0, 0]
        self.playedCards = []
        self.Order(self.dealer + 1)           
        self.currentEpoch = 0
        self.maximumEpoch = 0
        self.testingScores = []       
        self.testingCycleScoresTeam0 = []   
        self.testingCycleScoresTeam1 = []
        self.testingTeam0WinsThisCycle = 0     
        self.testingTeam1WinsThisCycle = 0
        self.testingWinRatioTeam0 = []
        self.testingTeam0IncrementalWins = []
        self.testingTeam1IncrementalWins = []
        self.testingTeam0TotalWins = []   
        self.testingTeam1TotalWins = []
        self.oldDiscountRate = y
 
    def SetPlayerBehaviour(self, playerID, behaviour):    
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
           
    def DealCards(self, d):    
        self.allPlayedCards = {}          
        self.roundScore = [0,0]  
        [p.hand.clear() for p in self.players]        
        for p in self.orderedPlayers:
            p.handSum = 0.
            p.hand = d.HandOutCards(self.orderedPlayers.index(p))
            p.feat = p.net.CreatePlayFeaturesVector(p, self, d)     
            for card in p.hand:
                p.handSum += card.value
                
    def PlayCards(self, d):     
        self.cardsOnTable = []
        self.leadingSuit = 'none'      
        for i,p in enumerate(self.orderedPlayers):
            if i == 0:
                self.playedCards  = [p.Play(self, d)]   
                self.leadingSuit  = self.playedCards[0].suit
            else:
                self.playedCards.append(p.Play(self, d))                 
            self.playedCards[i].whoPlayedMe = p.position             
        for c in self.playedCards:             
            tmp = self.playedCards.index(c)
            tmp = self.orderedPlayers[tmp]     
            tmp = self.players.index(tmp)      
            self.allPlayedCards[c] = tmp                
        self.playedTuples = [c.CardAsTuple() for c in self.playedCards]  
  
    def WhoWinsTrick(self, d):   
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
        self.roundScore[self.winnerPlayer.team] += trickPoints    
        for p in self.players:
            if p.team == self.winnerPlayer.team:
                p.reward = 1 
                if p == self.winnerPlayer:
                    p.reward += 0.2      
            else:
                p.reward = -1 
        self.Order(self.winnerPlayerID)                           
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
            if not self.players[0].testing:
                self.currentEpoch += 1         
        return self.winnerPlayer

    def checkEmptyHands(self): 
        if (len(self.players[0].hand) == 0 and len(self.players[1].hand) == 0
                and len(self.players[2].hand) == 0 and len(self.players[3].hand) == 0):
            return True
        return False

    def DoBackprop(self):
        with torch.no_grad(): 
            Plist = [f.feat.clone() for f in self.orderedPlayers]
            P = [Pl.clone() for Pl in Plist] 
            for feat_vec_index,feat_vec in enumerate(P):     
                for card_index,card in enumerate(feat_vec):         
                    if card_index == self.playedCards[feat_vec_index].index:        
                        feat_vec[card_index] = 0              
            new_state_Q = []
            updated_Q = [p.output.clone() for p in self.orderedPlayers]   
        for player_index,player in enumerate(self.orderedPlayers):                
            played_card_index = self.playedCards[player_index].index 
            new_state_Q.append(player.net(P[player_index]))                                                   
            updated_Q[player_index][played_card_index] += player.alpha * (player.reward + player.y * new_state_Q[player_index][player.FindAllowedMaximum(P[player_index])] - updated_Q[player_index][played_card_index])       
            player.computed_loss = player.loss(player.output, updated_Q[player_index])  
            player.opt.zero_grad()
            player.computed_loss.backward()                 
            player.opt.step()

    def calculateTestResults(self):
        array = np.array(self.testingScores, dtype='int')
        scoreTeam0 = 0
        scoreTeam1 = 0
        j = 0
        for i in range(len(self.testingScores)):
            if i % 8 == 7:
                scoreTeam1 += array[i,1]
                scoreTeam0 += array[i,0]
                j += 1
        self.testingCycleScoresTeam0.append(scoreTeam0/j)
        self.testingCycleScoresTeam1.append(scoreTeam1/j)
        winRatio = self.testingTeam0WinsThisCycle / (self.testingTeam0WinsThisCycle + self.testingTeam1WinsThisCycle)
        if not self.testingTeam0TotalWins or not self.testingTeam1TotalWins:  
            self.testingTeam0TotalWins.append(self.testingTeam0WinsThisCycle)
            self.testingTeam1TotalWins.append(self.testingTeam1WinsThisCycle)
        else: 
            self.testingTeam0TotalWins.append(self.testingTeam0WinsThisCycle + self.testingTeam0TotalWins[-1])
            self.testingTeam1TotalWins.append(self.testingTeam1WinsThisCycle + self.testingTeam1TotalWins[-1])
        self.testingTeam0WinsThisCycle = 0
        self.testingTeam1WinsThisCycle = 0
        self.testingWinRatioTeam0.append(winRatio)
