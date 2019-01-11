import random
from random import shuffle
import itertools
from typing import List, Any, Union, Tuple

import numpy as np


####################
# Creating the deck
####################
class Card:
    suits = None

    def __init__(self, values, suits, scores):
        self.values = values
        self.suits = suits
        self.scores = scores

    values = ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['spades', 'clubs', 'hearts', 'diamonds']


index = {'7': 1, '8': 2, '9': 3, '10': 4, 'J': 5, 'Q': 6, 'K': 7, 'A': 8}
scores = {'7': (0, 0), '8': (0, 0), '9': (0, 14), '10': (10, 10), 'J': (2, 20), 'Q': (3, 3), 'K': (4, 4), 'A': (11, 11)}
deck = list(itertools.product(Card.suits, Card.values))

###########
# information variables
##############
score = [0, 0]
playrounds = [0,0,0,0,0,0,0,0]
for i in range(0,8):
    playrounds[i] = list()
alwinner = [0, 0, 0, 0, 0, 0, 0, 0]
starttrick = [0, 0, 0, 0, 0, 0, 0, 0]

######################
# starting of the trick
#######################

# first player
scorestarstar = 0
startingplayer = 0
k = startingplayer
nrounds =8



# divide cards
random.shuffle(deck)
hands = list()
hands.append(deck[0:8])
hands.append(deck[8:16])
hands.append(deck[16:24])
hands.append(deck[24:32])
# setting trumpsuit
trumpsuit = 'hearts'
# table with the cards

for g in range(0, 4):

    cardsontable = list()
    # init playhands
    playhands = [0, 0, 0, 0]
    for i in range(0, 4):
        playhands[i] = list()
    # decision variable
    m = 0
    playhands[k] = hands[k]
    cardsontable.append(playhands[k][m])
    mr = hands[k].index(playhands[k][m])
    hands[k].pop(mr)
    otsuit = cardsontable[0][0]

    #########################
    # determining playable hands
    ##########################

    for j in range(0, 3):
        kstar = 1 + k
        if kstar == 4:
            k = 0
        else:
            k = kstar
        print(k)
        A = list()
        if otsuit == trumpsuit:
          for i in range(0, nrounds):
            if hands[k][i][0] == trumpsuit and index[hands[k][i][1]] > index[cardsontable[0][1]]:
              A.append(hands[k][i])
          if len(A) == 0:
            for i in range(0, nrounds):
              if hands[k][i][0] == trumpsuit:
                A.append(hands[k][i])
                print(2)
        else:
          for i in range(0, nrounds):
            if hands[k][i][0] == otsuit:
               A.append(hands[k][i])
               print(3)

          if len(A) == 0:
            for i in range(0, nrounds):
              if hands[k][i][0] == trumpsuit:
                A.append(hands[k][i])
                print(4)
        if len(A) == 0:
          A = hands[k]
          print(5)
        print(A)
        playhands[k] = A
        cardsontable.append(playhands[k][m])
        mr = hands[k].index(playhands[k][m])
        hands[k].pop(mr)

    indexstar = [0, 0, 0, 0]
    scorestar = 0
    for i in range(0, 4):
        if cardsontable[i][0] == trumpsuit:
            indexstar[i] = index[cardsontable[i][1]] * 20
            scorestar += scores[cardsontable[i][1]][1]
        elif cardsontable[i][0] == otsuit:
            indexstar[i] = index[cardsontable[i][1]]
            scorestar += scores[cardsontable[i][1]][0]
        else:
            indexstar[i] = 0
            scorestar += scores[cardsontable[i][1]][0]

    if nrounds == 1:
        scorestar += 10

    winner = np.argmax(indexstar)
    nrounds += -1
    k = winner
    playrounds[g] = cardsontable
    alwinner[g] = winner
    scorestarstar += scorestar
    if winner == 0 or winner == 2:
        score[0] += scorestar
    else:
        score[1] += scorestar


starttrick[1:8] = alwinner[0:7]

print(trumpsuit)
for i in range(0,8):
     print("{a} trick started {b}, winner  {c} ".format(a = playrounds[i], b = starttrick[i] , c = alwinner[i]))
print(score)


