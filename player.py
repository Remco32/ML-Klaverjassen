
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

        
    def Pop(self):
        self.subHand = [c for c in self.hand if c.isPlayable == True]
        popped = self.subHand.pop(rnd.randrange(0,len(self.subHand)))
        self.hand.remove(popped)
        return popped

        
\    # method to play a card from the hand.
    def Play(self, tab, d):  #tab is the table object for the rules, d is the deck object for the trump suit
        
        if tab.WhoPlays()[0] == self:    #if he's starting the trick
            for c in self.hand:
                c.isPlayable = True
            self.played = self.Pop()
            for c in self.hand:
                c.isPlayable = False
                        
        else:
            if tab.rules == 'Simple':
                self.SimplePlay(tab, d)
            elif tab.rules == 'Rotterdam':
                self.RotterdamPlay(tab, d)
            elif tab.rules == 'Amsterdam':
                self.AmsterdamPlay(tab, d)
            else:
                print('Unknown rules. Input either \'Simple\', \'Amsterdam\' or \'Rotterdam\'')
        return self.played             #self.played is assigned within each of the play methods below
    
    def SimplePlay(self, tab, d):
        playableCards = 0
        
        """
           New play routine which should fit better the machine learning implementation:
           the hand is scanned and if the cards are playable their data member .isPlayable
           becomes true, otherwise it is kept false. Then the card is played from the subset
           of cards which have .isPlayed == True. This last step will be changed when the
           RL algorithms are implemented, so for now it makes no difference, but having
           the playability as a property of the card is useful.
        """

        for c in self.hand:                     #first try to flag cards in suit as playable
            if c.suit == tab.leadingSuit:
                c.isPlayable = True
                playableCards += 1

        if playableCards == 0:                  #if none, then try flag trumps as playable (of course checking if playableCards == 0
            for c in self.hand:                 #also implies that trump != leading suit, so that cards are not counted twice)
                if c.suit == d.trumpSuit:
                    c.isPlayable = True
                    playableCards += 1

        if playableCards == 0:                  #otherwise flag any other card
            for c in self.hand:
                c.isPlayable = True 

        self.played = self.Pop()
        
        if self.hand != []:                       #after playing the card, all the others are flagged as unplayable before the next trick
            for c in self.hand:
                c.isPlayable = False

                
    def RotterdamPlay(self, tab, d):
        pass

    def AmsterdamPlay(self, tab, d):
        pass

