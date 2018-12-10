# PyTorch or TensorFlow for NN
# SkLearn for decision trees
#
#  Meeting Monday at 1pm room 228 (Whatsapp)
#
# Adjust the rules

import deck
import player
import table
import numpy as np


class Learn:

    def CreateTrumpFeaturesVector(self):
        """
        Needed features:
        - hand (32 binary features)
        - number of passes [0,1,2,3]
        - game score for my team
        - game score for opposite team
        - round number
        """
        pass

    def CreatePlayFeaturesVector(self, pl, tbl, dck):
        """
        Needed features:
        - hand (same as above)
        - which card were played and by who (32 numbers from 0 to 4: 0 not played, 1,2,3,4 number of player)
        - round scores (2 numbers)
        - trump (which suit [1,2,3,4], who chose it [1,2,3,4])  #for now since the trump choosing is not implemented it's just the suit
        (- round number and game scores) optional
        """
        tmp = []
        for c in dck.cards:      #deck.Deck.cards is an ordered list with card indexes from 0 to 31 
            if c in pl.hand:
                tmp.append(1)
                print(c.CardAsTuple())
            else:
                tmp.append(0)
                print(c.CardAsTuple())
        

        for c in dck.cards:
            if c in tbl.allPlayedCards.keys():
                tmp.append(tbl.allPlayedCards[c] + 1)
            else:
                tmp.append(0)

        

        [tmp.append(s) for s in tbl.roundScore]

        tmp.append(dck.suits.index(dck.trumpSuit) + 1) #code: 1,2,3,4 = d,c,h,s
        #tmp.append(who chose the trump)
                
        self.playFeaturesVector = np.array(tmp)     #since pytorch most likely wants numpy arrays as inputs;
        return self.playFeaturesVector
        
        
    
