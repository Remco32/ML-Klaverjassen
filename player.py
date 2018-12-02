#
#Class that represents players
#
#
import random as rnd
import deck

class Player:

    # player 0 and 2 are in team 0; player 1 and 3 are in team 1 
    def __init__(self, number):  #number is an integer from 0 to 3
        self.position = number
        self.hand = []

    #requests a hand of cards taken from the deck object
    #Only valid if the dealer is player 4
    #this method is obsolete; all cards are dealt at the same time
    #with the method table.dealCards()
    ##def requestHand(self):
    ##    self.hand = []
    ##    self.hand = deck.handOutCards(self.position)
    ##    deck.divideCards()
                    

    # method to play a card from the hand.
    # in this version, the player just plays a random card 
    def play(self):
        i = rnd.randrange(0,len(self.hand)) #chooses a random card
        played = self.hand.pop(i)       #plays chosen the card
        return played
    

    
        

