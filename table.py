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


class Table:

    def __init__(self, Round, rules, alpha, y):    #now all players have same learning rate and y. For exp. purposes it might be interesting to have different values
        self.playerID = 0,1,2,3             #the tuple's index is the player's name 
        self.cycleID = self.playerID * 2    #useful to cycle from player 4 to 1
        self.players  =[player.Player(i, alpha, y) for i in self.playerID]
        self.dealer   = rnd.choice(self.playerID)  #first dealer chosen randomly
        self.rules = rules
        self.roundScore = [0, 0]
        self.gameScore  = [0, 0]
        self.cardsOnTable = [-1, -1, -1]
        self.Order(self.dealer + 1)           #ordering the players with respect to the PLAYER STARTING THE TRICK (refer to cycleID, this
                                              #means 3+1=0)
        # For variables used in running the experiments
        self.currentEpoch = 0;
        self.maximumEpoch = 0;
       

                                                
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
        [p.hand.clear() for p in self.players]        
        for p in self.orderedPlayers:
            p.handSum = 0.
            p.hand = d.HandOutCards(self.orderedPlayers.index(p))
            p.feat = p.net.CreatePlayFeaturesVector(p, self, d)    #create the feature vector when the cards are dealt
            for card in p.hand:
                p.handSum += card.value
               

    def PlayCards(self, d):     #d is the deck object
        self.cardsOnTable = [-1, -1, -1]
        for i,p in enumerate(self.orderedPlayers):
            if i == 0:
                self.playedCards = [p.Play(self, d)]   #play the first card
                self.leadingSuit = self.playedCards[0].suit
            else:
                self.playedCards.append(p.Play(self, d))
                self.cardsOnTable[i - 1] = self.playedCards[i].index  #for the feature vector

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
                p.reward = 1.
                if p == self.winnerPlayer:
                    p.reward += 0.2      #reward good plays to single player
            else:
                p.reward = -1.
            p.rewardArray.append(p.reward)
            p.weightedRewardArray.append(p.reward / (d.maxCardValue*p.handSum))

        self.Order(self.winnerPlayerID)                           #the game starts from the trick winner

        if self.players[0].hand == []:
            self.roundScore[self.winnerPlayer.team] += 10
            
        
        return self.winnerPlayer


    def DoBackprop(self):
        with torch.no_grad():

            Plist = [f.feat.clone() for f in self.orderedPlayers]
            P = [torch.tensor(Pl, dtype=torch.float) for Pl in Plist]     #clone the feature vectors to delete played cards
                
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
            p.feat = P[i].clone()            #update the feature vector
            p.feat.requires_grad = True


    
    def whoWinsRound(self):
        pass


