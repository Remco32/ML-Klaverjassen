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

t = table.Table(1,'Simple')
d = deck.Deck()
l = learn.Learn()
##print('\nRound ', t.nRound) 
##print('PlayerIDs ', t.playerID)
##print('cycleIDs ', t.cycleID)
##print('Players ', t.players)
print('Dealer ', t.dealer)
##print('Ordered players ', t.orderedPlayers)   
print('First player ', t.WhoPlays()[1])

d.SetTrump(rnd.choice(d.suits))       #randomly chosen trump
d.DivideCards()                
##print('Divided cards', d.dividedCards)   ##method works
print('Trump suit', d.trumpSuit)


t.DealCards(d)
h = [p.hand for p in t.players]
for i in (0,1,2,3):
    ht = [c.CardAsTuple() for c in h[i]]
    print('Hand', i, ht)

#very simple routine in which only the values count, with no other rules
while t.players[0].hand != []:
    t.PlayCards(d)
    tmp = t.playedTuples           #to check if WhoWinsTrick works
    print('Played Cards', tmp)
    print('Winner player', t.WhoWinsTrick(d).position)
    fv = l.CreatePlayFeaturesVector(t.players[1], t, d)
    print(fv)
    print('Scores', t.roundScore)
