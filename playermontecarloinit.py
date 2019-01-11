# Monte Carlo
import numpy as np
import random
from random import shuffle
import itertools
import pickle


# creating shift function
# shift function is needed to obtain orderedplayrounds

def shift(l, n):    # shift function
    return l[n:] + l[:n]

#########################################
#                                       #
# ----------- Deck code --------------- #
#                                       #
#########################################
'''class Card:
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

'''
#########################################
#                                       #
# ----------- Inputs  ----------------- #
#                                       #
#########################################

#  Playrounds is a list with the played cards in each rounds
# So, playrounds[0] contains cards of the first round played
playrounds = playrounds

starttrick = starttrick          # contains a vector with the players who started the tricks.
                                  # So, starttrick[0] contains the player number of the player who started the first trick
handsplayer = hands[0]           # Contains the left over hand of the player

scores = scores                # denotes the current points earned.
G = 4 # len(playrounds)      # The number of round we are in right now, as an example I took the 4th round


#########################################
#                                       #
# ----------- Inputs  ----------------- #
#                                       #
#########################################

# Orderedpplayrounds is the same as playrounds but now the first element is always the cards which player 0 played

orderplayrounds = [ 0, 0, 0, 0, 0, 0, 0, 0 ]

for i in range(0, 8):
    orderplayrounds[i] = list()
    orderplayrounds[i].append(shift(playrounds[i], -starttrick[i]))


# Possuit creats a matrix which indicates if a player does not have suit

# 0:  if [1, 0, 0, 0] The player cannot have hearts
# 1:  if [0, 1, 0, 0] The player cannot have diamonds
# 2: if [0, 0, 1, 0] The player cannot have clubs
# 3: if [0, 0, 0, 1] The player cannot have spades

# example when an 1 is entered.
# lets say hearts is played and player 2 plays spades, then possuit[2, ] = [1, 0, 0, 0]

# we still need to add an other column for higher trump cards but that is a bit complicated and a rare situation
# so i can do it later

possuit = np.zeros((4, 5))

for g in range(0, G):
    suitstar = playrounds[g][0][0]
    for i in range(0, 4):
        if orderplayrounds[g][0][i][0] != suitstar:
            if suitstar == 'hearts':
                possuit[i, 0] = 1
            if suitstar == 'diamonds':
                possuit[i, 1] = 1
            if suitstar == 'clubs':
                possuit[i, 2] = 1
            if suitstar == 'spades':
                possuit[i, 3] = 1


# Now we create the adjusted deck which is the bases for the simulation.
# adjusted deck contains cards not owned by player 0, or played during the game.

adjdeck = [ 0, 0, 0, 0, 0, 0, 0, 0 ]
for i in range(0, 8):
    adjdeck[i] = list()
                                                # transforming the list to set apperantly works the best, so i did
adjdeck[0] = set(deck) - set(deck[0:8]) #handsplayer  # deck[0:8] is the hand of player 0, needs to be changed to handsplayer
for i in range(1, 8):                                  # deck is the set with all the cards
    adjdeck[i] = set(adjdeck[i-1])-set(orderplayrounds[i-1][0][1:4])  # Subtract the cards played

#########################################
#                                       #
# ------ Initialization --------------- #
#                                       #
#########################################
Q = 100                # number of simulations

mcscorearry = np.zeros((Q, 8))        # array containing the simulated scores
mcscore = [0, 0]                      # vector which will be added to the array
trumpsuit = 'hearts'          # fill in trumpsuit here
startingplayer = 0
k = startingplayer

suitclass = ['hearts', 'diamonds', 'clubs', 'spades']
mchandsuc = list()
mchands = list()

'''
mchands contains the list of unconditional possible hands for the players, 
unfortunatly it is quite tricky to directly divide the cards according to the 
conditions from possuit, so i created the unconditional list and then remove the 
not possible hands. 
'''

'''
Next we have the function which first determines mchandsuc and mchands, we shuffle cards which might be a bit overkill 
since the cards are shuffled again later on. 

Things to add, we might broaden the function such it removes the 'double' hands and we can let it run until we obtain a 
fixed number of mchands. 

'''

for j in range(0, 10000):
    a = 0
    mchandsuc.append([])
    mchandsuc[j].append(handsplayer)
    A = list(adjdeck[G])
    random.shuffle(A)
    for i in range(1, 4):
        mchandsuc[j].append(A[0:8 - G])
        for g in range(0, 8 - G):
            A.remove(A[0])
    for r in range(1, 4):
        for c in range(0, 4):
            if possuit[r, c] == 1:
                for z in range(0, 8 - G):
                    if mchandsuc[j][r][z][0] == suitclass[c]:
                        a = 1
    if a == 0:
        mchands.append(mchandsuc[j])


#########################################
#                                       #
# ------ Simulation of the game ------- #
#                                       #
#########################################

''''
In this section we run the simulations. I have had a lot of problems storing the mchands, since the list 
gets altered after one rounds is played. The solution i found is using pickle which solves the problem but is a 
bit slow. 


Further for simplicity I let Q be 100 this should be altered to length mchands or something. 

'''

storemchands = [i for i in mchands[:] ]

with open('storemchands', 'wb') as f:     # so here is the storing
    pickle.dump(storemchands, f)


for i in range(0, Q):                       # here we start running the game 100 times.
    with open('storemchands', 'rb') as f:
        storemchands = pickle.load(f)

    nrounds = 8 - G                 # denotes the numbers of rounds left to play
    q = i
    if i in range(0, 25):           # f denotes the card used for simulation, now I see it only works for 4 cards,
        f = 0                       # I might try to improve it
    if i in range(25, 50):
        f = 1
    if i in range(50, 75):
        f = 2
    if i in range(75, 100):
        f = 3

    mcscore = [0, 0]            # keeps track of the scores
    mchands = storemchands
    firstcard = mchands[q][0][f]
    mchands[q][0].remove(firstcard)
    random.shuffle(mchands[q][0])
    mchands[q][0].insert(0, firstcard)  # This part I reorder the hand such card (index) f is first
                                         # and the others are random
    mchandsstar = list()
    mchandsstar = [i for i in mchands[0]] # this is a bit useless but i was trying to fix storage problem

    for g in range(G, 8):                        # and now the game begins
        cardsontable = list()
        playhands = [0, 0, 0, 0]
        for i in range(0, 4):
            playhands[i] = list()
        m = 0                                   # always play first cards in the hand. (hands are always different
        playhands[k] = mchandsstar[:][k]
        cardsontable.append(playhands[k][m])            # denotes the cards played during the round
        mr = mchandsstar[:][k].index(playhands[k][m])   # index of card played
        mchandsstar[:][k].pop(mr)                        # remove the cards played
        otsuit = cardsontable[0][0]                      # determines suit which is played
                                                         # Move of the starting player
        #############################
        # determining playable hands#
        #############################

        for j in range(0, 3):      # k denotes the player who plays
            kstar = 1 + k
            if kstar == 4:
                k = 0
            else:
                k = kstar

            A = list()
            if otsuit == trumpsuit:                     # conditions when trump suit is played
              for i in range(0, nrounds):
                if mchandsstar[:][k][i][0] == trumpsuit and index[mchandsstar[k][i][1]] > index[cardsontable[0][1]]: # for index look at the deck. it is basically the order of cards.
                  A.append(mchandsstar[:][k][i])
              if len(A) == 0:
                for i in range(0, nrounds):
                  if mchandsstar[:][k][i][0] == trumpsuit:
                    A.append(mchandsstar[:][k][i])
            else:    # conditions when no trump card is played
              for i in range(0, nrounds):
                if mchandsstar[:][k][i][0] == otsuit:
                    A.append(mchandsstar[:][k][i])
               if len(A) == 0:
                for i in range(0, nrounds):
                  if mchandsstar[:][k][i][0] == trumpsuit:
                    A.append(mchandsstar[:][k][i])

            if len(A) == 0:
              A = mchandsstar[:][k]

            playhands[k] = A
            cardsontable.append(playhands[k][m])
            mr = mchandsstar[:][k].index(playhands[k][m])
            mchandsstar[:][k].pop(mr)

        indexstar = [0, 0, 0, 0]
        scorestar = 0
        for i in range(0, 4):                           # determines the scores
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

        winner = np.argmax(indexstar) # gives the winner
        nrounds += -1                 # next round
        k = winner
        playrounds[g] = cardsontable
        if winner == 0 or winner == 2:
            mcscore[0] += scorestar
        else:
            mcscore[1] += scorestar



    print(trumpsuit)  # Overview of the game played.
    for i in range(0,8):
         print("{a} trick started {b}, winner  {c} ".format(a = playrounds[i], b = starttrick[i] , c = alwinner[i]))
    print(mcscore)

    print(sum(mcscore + score))
    mcscorearry[q, 2*f:2*f+2] = [mcscore[0] + score[0], mcscore[1] + score[1]]   # array which stores all the scores


    print(mchandsstar[0])
    print(mchands[0][0])


print(sum(mcscorearry[:,0])/25)  # score if first cards of the hand is played
print(sum(mcscorearry[:,2])/25)  # score if the second ....
print(sum(mcscorearry[:,4])/25)  # .....
print(sum(mcscorearry[:,6])/25)

# best cardlist contains the scores for all the cards
bestcardlist = [sum(mcscorearry[:,0])/25, sum(mcscorearry[:,2])/25, sum(mcscorearry[:,4])/25, sum(mcscorearry[:,6])/25]
bestcard = np.argmax(bestcardlist) # this is the output, hence the card that should be played according to the algorithm
print(hands[0][bestcard])



