
#Class that represents players
#
#I am implementing simple rules to begin with; they are as simple as
# - if you're starting the trick, play any card; otherwise:
# - play in suit if possible
# - if not possible play a trump if possible
# - if not possible play any card
#
#I suggest we implement Rotterdam rules after because they seem easier according
#to this link https://www.pagat.com/jass/klaverjassen.html
#We should then implement Amsterdam rules as well, switching between
#the rules with the table.Table.rules variable
#
#The lines marked by ### are the lines where machine learning algoriths
#will take place
#

import random as rnd
import deck
import table

class Player:

    # player 0 and 2 are in team 0; player 1 and 3 are in team 1 
    def __init__(self, number):  #number is an integer from 0 to 3
        self.position = number
        self.hand = []
        if number % 2 == 0:
            self.team = 0
        else:
            self.team = 1

    def InHand(self, suit):
        self.subHand = [c for c in self.hand if c.suit == suit]   #build a subhand for the passed suit
        if self.subHand != []:
            return True
        else:
            return False

    # method to play a card from the hand.
    def Play(self, tab, d):  #tab is the table object for the rules, d is the deck object for the trump suit
        
        if tab.WhoPlays()[1] == self:       #check if he's the first to play
            cardID = rnd.randrange(0, len(self.hand))   ###
            self.played = self.hand.pop(cardID)
        
        else:
            if tab.rules == 'Simple':
                self.SimplePlay(tab, d)
            elif tab.rules == 'Rotterdam':
                self.RotterdamPlay(tab, d)
            elif tab.rules == 'Amsterdam':
                self.AmsterdamPlay(tab, d)
            else:
                print('Unknown rules. Input either Amsterdam or Rotterdam')
        return self.played
    
    def SimplePlay(self, tab, d):
         if self.InHand(tab.leadingSuit) == True:             #play in suit if possible
             cardID = rnd.randrange(0, len(self.subHand))     ###
             self.played = self.subHand.pop(cardID)           #card to be played
             self.hand.remove(self.played)                    #remove the card from the hand (since subHand and hand are different lists)
         elif self.InHand(d.trumpSuit) == True:
             cardID = rnd.randrange(0, len(self.subHand))     ###
             self.played = self.subHand.pop(cardID)
             self.hand.remove(self.played)
         else:
             cardID = rnd.randrange(0, len(self.hand))        ###
             self.played = self.hand.pop(cardID)
    
    def RotterdamPlay(self, tab, d):
        pass

    def AmsterdamPlay(self, tab, d):
        pass

