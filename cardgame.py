# Complete deck including trump and shuffle routine.
#
#

# dictionary for the deck including suit, names and values
deckDict = {
    'd': {'K': 13, 'Q': 12, 'J': 11, '10': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2, 'A': 1},
    'c': {'K': 13, 'Q': 12, 'J': 11, '10': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2, 'A': 1},
    'h': {'K': 13, 'Q': 12, 'J': 11, '10': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2, 'A': 1},
    's': {'K': 13, 'Q': 12, 'J': 11, '10': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2, 'A': 1}
    }


# class to represent a card in the game
##the value attribute is kept separate cause it is conceptually different:
##while the suit and the name ("queen of spades", "ace of hearts") are
##printed on the card and are immutable, the value of each card changes
##depending on the game and even on each particular round (trump suit
##having higher values than usual)
class Card:

    def __init__(self, suit, name):
        self.suit = suit
        self.name = name

    def SetValue(self, value):
        self.value = value


# create a deck as a list of instances of the class (a list of cards)
deck = [Card(s, n) for s in deckDict.keys() for n in deckDict[s].keys()]

# declare the trump suit; now it's hearts
trump = 'h'
maxValue = deckDict['d']['K']  # the highest non-trump value

# adding the maxValue to each trump card's value makes it
# more valuable than any other non-trump card
[card.SetValue(deckDict[card.suit][card.name]) for card in deck]
[card.SetValue(deckDict[card.suit][card.name] + maxValue) for card in deck if card.suit == trump]
print("Deck:")
[print(card.name, card.suit, card.value) for card in deck]

# now to the deck shuffling
import random as rnd

rnd.shuffle(deck)
print("Shuffled deck:")
print('\n')
[print(card.name, card.suit, card.value) for card in deck]