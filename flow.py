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

t = table.Table(1,'Rotterdam')
d = deck.Deck()
print('\nRound ', t.nRound)
print('PlayerIDs ', t.playerID)
print('cycleIDs ', t.cycleID)
print('Players ', t.players)
print('Dealer ', t.dealer)
print('Ordered players ', t.orderedPlayers)   
print('First player ', t.WhoPlays())

d.SetTrump(rnd.choice(d.suits))       #randomly chosen trump
d.DivideCards()
print('Divided cards ', d.dividedCards)

t.DealCards(d)
[print('Hand ', p.hand) for p in t.players]

#very simple routine in which only the values count, with no other rules
while t.players[0].hand != []:
    t.PlayCards()
    tmp = t.playedTuples           #to check if WhoWinsTrick works
    print('Played Cards ', tmp)
    print('Winner card ', t.WhoWinsTrick().CardAsTuple())
