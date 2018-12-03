
#Class that represents players
#
#I am implementing Rotterdam rules because they seem easier according
#to this link https://www.pagat.com/jass/klaverjassen.html
#We should then implement Amsterdam rules as well, switching between
#the rules with the table.Table variable self.rules
#
import random as rnd
import deck
import table

class Player:

    # player 0 and 2 are in team 0; player 1 and 3 are in team 1 
    def __init__(self, number):  #number is an integer from 0 to 3
        self.position = number
        self.hand = []
                    

    # method to play a card from the hand.
    # in this version, the player just plays a random card 
    def Play(self, tab):  #tab is the table object for the rules
        
       # if tab.WhoPlays()[1] == self:       #check if he's the first to play
        cardID = rnd.randrange(len(self.hand))
        played = self.hand.pop(cardID)
        
       # else:
        #    if tab.rules == 'Rotterdam':
         #       
          #  elif tab.rules == 'Amsterdam':
           #     pass
            #else:
             #   print('Unknown rules. Input either Amsterdam or Rotterdam')
        return played
    

    
        

