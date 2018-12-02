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

t = table.Table(1)
print('\nRound ', t.nRound)
print('PlayerIDs ', t.playerID)
print('cycleIDs ', t.cycleID)
print('Players ', t.players)
print('Dealer ', t.dealer)
print('Ordered players ', t.orderedPlayers)   
print('First player ', t.whoPlays())

deck.divideCards()
print('Divided cards ', deck.dividedCards)

t.dealCards()
[print('Hand ', p.hand) for p in t.players]

#very simple routine in which only the values count, with no other rules
while t.players[0].hand != []:
    a = t.playCards()
    [print('Played Cards ', a)]
    print('Winner card ', t.whoWinsTable())
