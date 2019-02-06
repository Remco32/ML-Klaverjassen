import random as rnd
import math

#TODO write a main class for handling all sorts of exploration strategies, i.e. you only have to call the main function of this file, not a sub function

# NOTE: Please pass on the subset of cards in the hand that are legal to play.


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
        return rnd.choice(playerHand) # pick random option
    else:
        return greedy(greedyAction)
    
def greedy(greedyAction):
    return greedyAction

#Max-Boltzmann takes a distribution of all possible actions, and assigns probabilities to these actions. This stops the
# top-most action to be overly-dominant, which isn't a good exploration strategy to begin with.

#See also http://fse.studenttheses.ub.rug.nl/15450/ and https://www.cs.mcgill.ca/~cs526/roger.pdf for pseudocode and the Boltzmann equation

def boltzmannExploration(explorationRate, playerHand, outputLayer):
    randomValue = rnd.uniform(0, 1) #between 0 and 1, float
    if randomValue < explorationRate:
        return rnd.choice(playerHand) # pick random option #TODO could pick illegal action
    else:
        boltzmann(outputLayer)

def boltzmann(outputLayer):

    #TODO make temperature dynamic
    temperature = 0.5

    #Calculate denominator of the Boltzmann function
    for i in range(outputLayer): #TODO check for off-by-one error
        denominator =+ math.exp(outputLayer[i] * temperature^(-1))

    probabilities = [] #empty list
    for i in range(outputLayer): #TODO check for off-by-one error
        probabilities.append( math.exp(outputLayer[i] * temperature^(-1)) / denominator )

    #Accumulate the probabilities



#Testing
testOutputLayer = []