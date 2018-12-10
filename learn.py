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
                # print("LOOK HERE DUMMY!: " + str(c.CardAsTuple()))
            else:
                tmp.append(0)
                # print("LOOK HERE DUMMY!: " + str(c.CardAsTuple()))

        # Generate part of feature vector for keeping track of which cards are played
        for c in dck.cards:
            if c in tbl.allPlayedCards.keys():
                tmp.append(tbl.allPlayedCards[c] + 1)
            else:
                tmp.append(0)

        # Current game scores for team 0 and team 1 #TODO sorta untested!
        [tmp.append(s) for s in tbl.gameScore]

        # Round number #TODO sorta untested!
        tmp.append(tbl.nRound)



        self.playFeaturesVector = np.array(tmp)  # since pytorch most likely wants numpy arrays as inputs;
        return self.playFeaturesVector


    # Feature vector looks like this: #TODO finish for easier reading fo the code
    # First 32 values correspond to the hand of the player
    # Then 32 values corresponding to which cards have been played in this round
    ## Each of these 32 values or keeping track of the cards are set up like this:
    ## 4 sets of cards, in order of suits [d, c, h, s]
    ## First 8 values are the hearts, in the order [A, 10, K, Q, J, 9, 8 ,7]
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
        # Generate part of feature vector for the player's hand
        tmp = []
        for c in dck.cards:      #deck.Deck.cards is an ordered list with card indexes from 0 to 31 
            if c in pl.hand:
                tmp.append(1)
                #print("LOOK HERE DUMMY!: " + str(c.CardAsTuple()))
            else:
                tmp.append(0)
                #print("LOOK HERE DUMMY!: " + str(c.CardAsTuple()))
        
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
        #tmp.append(who chose the trump) #TODO implement reading the actual trump value
                
        self.playFeaturesVector = np.array(tmp)     #since pytorch most likely wants numpy arrays as inputs;
        return self.playFeaturesVector
        
        
    
