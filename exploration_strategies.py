import random as rnd
import math

def diminishingEpsilonGreedy(greedyAction, epsilon, playerHand, currentEpoch, totalEpocs):
    # Set current exploration rate
    minimumExplorationRate = 0.01
    explorationRate = epsilon * (1 - (currentEpoch / totalEpocs))
    # Check for minimum limit exploration rate
    if explorationRate < minimumExplorationRate:
        explorationRate = minimumExplorationRate
    # Rest of the algorithm is essentially epsilon-greedy
    return epsilonGreedy(greedyAction, explorationRate, playerHand)

def epsilonGreedy(greedyAction, explorationRate, playerHand):
    randomValue = rnd.uniform(0, 1) 
    if randomValue < explorationRate:
        return rnd.choice(playerHand) 
    else:
        return greedy(greedyAction)
    
def greedy(greedyAction):
    return greedyAction
