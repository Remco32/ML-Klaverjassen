import numpy as np
import deck
import table
import time


class Simulation:
    '''Performs a Monte Carlo simulation for one of the players.

    Parameters: Deck, Table, Player ID (default 0), number of simulations (default 10000)'''

    def __init__(self, d, tab, playerID = 0, simulations = 10000):
        self.playerID = playerID
        self.simulations = simulations

    def storeHands(self, tab):
        '''Stores the existing hands so they don't get lost when overwritten afterwards'''
        self.storedHand = []
        for i in range(4):
            self.storedHand.append(tab.players[i].hand)

    def returnHands(self, tab):
        '''Returns the stored hands so the original ones don't remain overwritten'''
        for i in range(4):
            tab.players[i].hand = self.storedHand[i]

    def setPlayer(self, playerID):
        '''Adjusts the player on which the simulation is performed'''
        self.playerID = playerID

    def setupSimulation(self, tab):
        '''Sets up the simulation at the start of a round.'''
        self.storeHands(tab)
        self.adjustedDeck = np.zeros(32) 
        self.handScores = [[] for i in range(len(tab.players[self.playerID].hand))] # Clears the scores from the hands.
        self.means = [0 for i in range(len(tab.players[self.playerID].hand))] # Stores the average means based upon the index.

        if len(tab.players[0].hand) == 8: # Resets the scores at the start of the round.
            tab.roundScore = [0, 0]

        for p in tab.playerID: # Sets the selected player as Montecarlo, rest at random for faster simulation (can use a Network in theory).
            tab.SetPlayerBehaviour(p, "MonteCarlo" if p == self.playerID else "Random")

    def adjustDeck(self, d, tab):
        '''Adjusts the deck based on the cards that have been played and in hand.
        This information is then put in a vector with a binary value for every single card (1 = played or in hand, 0 = to be played).'''

        for c in d.cards:
            id = getattr(c, "index")

            if c in tab.players[self.playerID].hand:
                self.adjustedDeck[id] = 1

            if c in tab.playedCards:
                self.adjustedDeck[id] = 1 

    def reassignCards(self, d, tab):
        '''Reassigns the cards to the other players to random cards from the adjusted deck.'''
        # Before starting we must clear the hands and create the pool.
        tab.players[1].hand, tab.players[2].hand, tab.players[3].hand, self.cardPool = ([] for i in range(4)) 
        self.cardPool = [card for card, idx in enumerate(self.adjustedDeck) if idx == 0] # Now, first we extract the not used cards
        np.random.shuffle(self.cardPool) # Then we shuffle the deck.

        # Assigning hands to other players
        m = 0
        for p in tab.playerID:
            if p != self.playerID:
                for card in range(len(tab.players[self.playerID].hand)):
                    tab.players[p].hand.append(d.cards[self.cardPool[card + m * len(tab.players[self.playerID].hand)]])
                m += 1

    def checkTrump(self, d, tab):
        '''Ensures that a trump card is played if there's still one in hand, even if that one does not have the best average score.
        If there are multiple trump cards in hand, it will use the one with the best average. '''
        self.bestCardIdx = 0
        self.hasATrump = False

        # Sorting the means from max to min
        self.sortedMeans = sorted(self.means, reverse=True)

        # Searching for trumps in hand
        for card in tab.players[self.playerID].hand:
            if card.suit == d.trumpSuit:
                self.hasATrump = True

        # If so, go through the hand and keep trying until we hit the trump with the best average score.
        if self.hasATrump:
            while self.bestCard.suit != d.trumpSuit:
                self.bestCardIdx += 1
                self.bestCard = tab.players[self.playerID].hand[self.means.index(
                    self.sortedMeans[self.bestCardIdx])]

    def run(self, d, tab):
        '''Runs the simulation.'''
        self.setupSimulation(tab)
        self.adjustDeck(d, tab)
        self.winScores, self.winCards = [], []
        self.saveScore = tab.roundScore

        for s in range(self.simulations):
            self.reassignCards(d, tab)
            tab.roundScore = [0, 0]
            tab.PlayCards(d)
            winner = tab.WhoWinsTrick(d)
            playerScore = tab.roundScore[0 if self.playerID in (0, 2) else 1] 
            self.handScores[tab.players[self.playerID].hand.index(tab.players[self.playerID].played)].append(playerScore)

        tab.roundScore = self.saveScore                     # Reassign the old score from the previous trick.    
        
        for card in range(len(self.handScores)):            # Calculating the mean score for every card.
            self.means[card] = np.mean(self.handScores[card])

        self.bestCard = tab.players[self.playerID].hand[self.means.index(np.max(self.means))] # Assign the card with the best average.
        self.checkTrump(d, tab)        
        self.returnHands(tab)                               # We give the original hands backs to the players.
        tab.players[self.playerID].played = self.bestCard
        tab.SetPlayerBehaviour(self.playerID, "MonteCarlo2") # Setting to the second mode so it actually discards the card away.
