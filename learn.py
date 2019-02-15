
import deck
import player
import table
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):   
    def __init__(self, nFeat):
        super(Net, self).__init__()
        self.conn1 = nn.Linear(nFeat, 150)            
        self.conn5 = nn.Linear(150, 32)
    def forward(self, tensor):
        tensor = torch.sigmoid(self.conn1(tensor))
        tensor = self.conn5(tensor)
        return tensor
    
    def CreatePlayFeaturesVector(self, pl, tbl, dck):
        tmp = []
        # Player's hand
        for c in dck.cards:     
            if c in pl.hand:
                tmp.append(1)
            else:
                tmp.append(0)       
        # Generate part of feature vector for keeping track of which cards have been played in the round
        for c in dck.cards:
            if c in tbl.allPlayedCards.keys():
                tmp.append(tbl.allPlayedCards[c] + 1)
            else:
                tmp.append(0)
        # Generate part of feature vector for keeping track of which cards are currently on the table
        if tbl.playedCards == []:
            for i in range(32):
                tmp.append(0)
        else:
            for c in dck.cards:
                if c in tbl.playedCards:
                    tmp.append(c.whoPlayedMe + 1) 
                else:
                    tmp.append(0)
       # Which card is the trump?
        tmp.append(dck.suits.index(dck.trumpSuit) + 1)
        self.playFeaturesVector = torch.tensor(tmp, dtype=torch.float, requires_grad=True)    
        self.nFeatures = len(tmp)
        return self.playFeaturesVector

    def UpdateFeatureVectors(self, pl, tbl, dck):
        self.CreatePlayFeaturesVector(pl, tbl, dck)
        return self.playFeaturesVector, 0
      
