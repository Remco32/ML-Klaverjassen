# PyTorch for NN
# SkLearn for decision trees
#
#
# Adjust the rules

import deck
import player
import table
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
This class handles all the neural network business, including creating the feature vectors.
The features vector is a data attribute of the class; if the two feature creating methods
are called subsequentially, the second feature vector will override the first one: so for
each network one must define a different Net object.
"""


class Net(nn.Module):


    ###################################
    #         Neural network          #
    ###################################

    
    def __init__(self, nFeat):
        super(Net, self).__init__()
        self.conn1 = nn.Linear(nFeat, 50)             #since each layer will take a lin combination of the previous layer's nodes
        self.conn2 = nn.Linear(50,50)
        self.conn3 = nn.Linear(50, 32)                 #32 is the deck size

    def forward(self, tensor):
        tensor = torch.sigmoid(self.conn1(tensor))   #pass the input layer to the first hidden layer through connection 1 defined in __init__
        tensor = torch.sigmoid(self.conn2(tensor))   #pass the first hidden layer to the second hidden layer through connection 2
        tensor = self.conn3(tensor)    #pass the second hidden layer to the output layer through connection 3 without any activation function
        return tensor






    ##################################################
    #             Feature creating methods           #
    ##################################################
    
    def CreateTrumpFeaturesVector(self, pl, tbl, dck):
        """
        Needed features:
        - hand (32 binary features)
        - number of passes [0,1,2,3]
        - game score for my team
        - game score for opposite team
        - round number
        """

        tmp = []
        # Generate part of feature vector for the player's hand
        tmp = []
        for c in dck.cards:  # deck.Deck.cards is an ordered list with card indexes from 0 to 31
            if c in pl.hand:
                tmp.append(1)
            else:
                tmp.append(0)

        # Generate part of feature vector for keeping track of which cards are played
        for c in dck.cards:
            if c in tbl.allPlayedCards.keys():
                tmp.append(tbl.allPlayedCards[c] + 1)
            else:
                tmp.append(0)

        # Current game scores for team 0 and team 1
        [tmp.append(s) for s in tbl.gameScore]

        # Round number
        tmp.append(tbl.nRound)



        self.featuresVector = torch.tensor(tmp, dtype=torch.float, requires_grad=True)  # since pytorch wants tensors as inputs;
        self.nFeatures = len(tmp)
        return self.featuresVector

    

    # Feature vector looks like this: #TODO finish for easier reading fo the code
    # First 32 values correspond to the hand of the player
    # Then 32 values corresponding to which cards have been played in this round
    ## Each of these 32 values or keeping track of the cards are set up like this:
    ## 4 sets of cards, in order of suits [d, c, h, s]
    ## First 8 values are the diamonds, in the order [A, 10, K, Q, J, 9, 8 ,7]
    # 2 features for the round scores for the teams
    # 1 feature for the trump suit
    # 3 features for the cards currently on the table


    
    def CreatePlayFeaturesVector(self, pl, tbl, dck):
        """
        This will have to be called after every trick is played, to update the features.
        Needed features:
        - hand (same as above)
        - which card were played and by who (32 numbers from 0 to 4: 0 not played, 1,2,3,4 number of player)
        - round scores (2 numbers)
        - trump (which suit [1,2,3,4], who chose it [1,2,3,4])  #for now since the trump choosing is not implemented it's just the suit
        - cards currently on the table
        (- round number and game scores) optional
        """
        tmp = []
        # Generate part of feature vector for the player's hand
        tmp = []
        for c in dck.cards:      #deck.Deck.cards is an ordered list with card indexes from 0 to 31 
            if c in pl.hand:
                tmp.append(1)
            else:
                tmp.append(0)
        
        # Generate part of feature vector for keeping track of which cards are played
        for c in dck.cards:
            if c in tbl.allPlayedCards.keys():
                tmp.append(tbl.allPlayedCards[c] + 1)
            else:
                tmp.append(0)

        
        # Current round scores for team 0 and team 1
        [tmp.append(s) for s in tbl.roundScore]

        # Which card is the trump?
        tmp.append(dck.suits.index(dck.trumpSuit) + 1) #code: 1,2,3,4 = d,c,h,s

        for _ in tbl.cardsOnTable:
            tmp.append(_)                #this is a list implemented in table.py that contains
                                         # -1 if some cards haven't been played, and the index otherwise
        #tmp.append(who chose the trump) #TODO implement reading the actual trump value
                
        self.featuresVector = torch.tensor(tmp, dtype=torch.float, requires_grad=True)     #since pytorch wants tensors as inputs;
        self.nFeatures = len(tmp)
        return self.featuresVector

