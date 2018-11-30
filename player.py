#
#Class that represents players
#
#
import random as rnd
import deck

class Player:

    # player 4 is the dealer; player 1 receives the first card, and is
    # the one who starts the game.
    # player 1 and 3 are in team 1; player 2 and 4 are in team 2 
    def __init__(self, number):  
        self.position = number

    #requests a hand of cards taken from the deck object
    def requestHand(self):
        self.hand = []
        self.hand = deck.handOutCards(self.position)
        deck.divideCards()
                    

    # method to play a card from the hand.
    # in this version, the player just plays a random card 
    def play(self):
        i = rnd.randrange(0,len(self.hand)) #chooses a random card
        played = self.hand.pop(i)       #plays chosen the card
        return played

    
        

