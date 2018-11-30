#
#Class that represents players
#
#
import random as rnd

class Player:

    # player 4 is the dealer; player 1 receives the first card, and is
    # the one who starts the game.
    # player 1 and 3 are in team 1; player 2 and 4 are in team 2 
    def __init__(self, number):  
        self.position = number

    #requests a hand of cards taken from the deck object
    def requestHand(self, #deck object):
        self.hand = #some deck method to hand out the cards
                    

    # method to play a card from the hand.
    # in this version, the player just plays a random card 
    def play(self):
        rnd.randrange(1,len(self.hand)) #chooses a random card
        played = self.hand.pop(i)       #plays chosen the card
        return played

    
        

