###########################################################################
# Full program flow concept
#
# x- initialise a table object: this also creates the 4 player objects
# x- divide the deck (this includes deck shuffling, see deck.py)
# x- deal the cards
# x- players play the cards
#  - count the point for the table
#  - after 8 table plays, count the points for the round
#  - update the dealer
#  - after 16 rounds, count the points for the game
#
###########################################################################

import table
import deck
import random as rnd
import learn

epochs, printEpoch, saveEpoch = 100, 1, 10

"""
STEPS TO UNDERTAKE TO PROPERLY RUN THE PROGRAM:

Initialise the table object: it automatically initialises the player objects, which in turn initialise the neural networks.
Initialise the deck, which in turn already shuffles.
Set the trump suit.
Divide the cards.
Deal the cards to the players; this also creates the starting feature vectors for each player once they receive their hand.
Print the hands of the players to check the game from command line.

A round in the game: 
until the players have cards in the hands, they play a card. If they select an illegal move, backprop is done internally until they select a legal one.
After all players have played their cards, a control printout is performed.
The trick winner is determined, rewards are assigned and then the table object performs the backprop for each player.

Multiple rounds in the game to properly train the network. Same as above, repeated for a number of epochs, with some more control printouts.
"""

for currentEpoch in range(epochs):
    
    d = deck.Deck()
    t = table.Table(16, 'Simple', 0.01, 0.9)
    
    d.SetTrump(rnd.choice(d.suits))  
    d.DivideCards()                
    print('Trump suit', d.trumpSuit)
    t.DealCards(d)
    h = [p.hand for p in t.players]
    for i in (0,1,2,3):
        ht = [c.CardAsTuple() for c in h[i]]
        print('Hand', i, ht)
    
    while t.players[0].hand != []:
        t.PlayCards(d)
        tmp = t.playedTuples       
        print('Played Cards', tmp)
        winner = t.WhoWinsTrick(d)
        t.DoBackprop()
        input("\nContinue?")

    if currentEpoch % printEpoch == 0: print("Epoch {} of {}".format(currentEpoch, epochs))
    if currentEpoch % saveEpoch  == 0:
        pass #save the network parameters; write a method in table.py


print("Training completed succesfully!")
