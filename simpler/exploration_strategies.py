import random as rnd

#TODO write a main class for handling all sorts of exploration strategies, i.e. you only have to call the main function of this file, not a sub function

def diminishingEpsilonGreedy(greedyAction, epsilon, playerHand, currentEpoch, totalEpocs):
# Diminishing epsilon-Greedy
    # Set current exploration rate
    minimumExplorationRate = 0.01
    explorationRate = epsilon * (1 - (currentEpoch / totalEpocs))

    # Check for minimum limit exploration rate
    if explorationRate < minimumExplorationRate:
        explorationRate = minimumExplorationRate

    # Rest of the algorithm is essentially epsilon-greedy
    return epsilonGreedy(greedyAction, explorationRate, playerHand)


def epsilonGreedy(greedyAction, explorationRate, playerHand):
    randomValue = rnd.uniform(0, 1) #between 0 and 1, float
    if randomValue < explorationRate:
        return rnd.choice(playerHand)
    else:
        return greedy(greedyAction)
    
#TODO: Placeholder. We can implement the function that picks the best legal move here
def greedy(greedyAction):
    return greedyAction

