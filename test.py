"""
To run single printouts

"""

import deck
import player
import table
import learn
import torch
import numpy as np

d = deck.Deck()
d.SetTrump('c')
d.DivideCards()
t = table.Table(16, 0.01, 0.9)
t.DealCards(d)

t.currentEpoch = 1
t.maximumEpoch = 2

t.PlayCards(d)
print('played tuples', t.playedTuples)
print('player 0 features', t.players[0].feat)
t.PlayCards(d)
print('played tuples', t.playedTuples)
print('player 0 features', t.players[0].feat)

while t.players[0].hand != []:
    t.PlayCards(d)
    winner = t.WhoWinsTrick(d)
    if t.players[0].testing == False:
        t.DoBackprop()
    elif t.players[0].testing == True:
        t.testingScores.append(t.roundScore.copy())
