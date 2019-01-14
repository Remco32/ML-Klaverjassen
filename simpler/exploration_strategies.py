import random as rnd

#TODO write a main class for handling all sorts of exploration strategies, i.e. you only have to call the main function of this file, not a sub function

def diminishingEpsilonGreedy(greedyActionIndex, epsilon, sizeOfPlayerHand, currentEpoch, totalEpocs):
# Diminishing epsilon-Greedy

    # Set current exploration rate
    minimumExplorationRate = 0.01

    explorationRate = epsilon * (1 - (currentEpoch / totalEpocs))

    # Check for minimum limit exploration rate

    if explorationRate < minimumExplorationRate:
        explorationRate = minimumExplorationRate

    # Rest of the algorithm is essentially epsilon-greedy
    return epsilonGreedy(greedyActionIndex, explorationRate, sizeOfPlayerHand)


def epsilonGreedy(greedyActionIndex, explorationRate, sizeOfPlayerHand):
    random = rnd.uniform(0, 1)
    if random < explorationRate:
        return rnd.choice(sizeOfPlayerHand) #size instead of actual hand so so we can return an index
    else:
        return greedyActionIndex