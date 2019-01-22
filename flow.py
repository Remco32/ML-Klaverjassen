import table
import deck
import random as rnd
import learn
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#trainingEpochs, printEpoch, saveEpoch = 1000, 100, 10
printEpoch, saveEpoch = 10, 10


# Load parameters?
#loadP = 0 #TODO check if loading works like it should
#Create folder if needed
#SAVEFOLDER = os.path.dirname(__file__) + '/weights/'

#if not os.path.exists(SAVEFOLDER):
#    os.makedirs(SAVEFOLDER)

#SAVEFILELIST = ['player' + str(i) + '_weights.pth' for i in range(4)]
#print(SAVEFILELIST)

"""
STEPS TO UNDERTAKE TO PROPERLY RUN THE PROGRAM:

######################
Initialise the table object: it automatically initialises the player objects, which in turn initialise the neural networks.
Initialise the deck, which in turn already shuffles.


######################
A round in the game: 

Set the trump suit.
Divide the cards.
Deal the cards to the players; this also creates the starting feature vectors for each player once they receive their hand.
OPT: Print the hands of the players to check the game from command line.
Until the players have cards in the hands, they play a (legal) card.
OPT: After all players have played their cards, a control printout is performed.
The trick winner is determined, rewards are assigned and then the table object performs the backprop for each player.

####################
Multiple rounds in the game to properly train the network. Same as above, repeated for a number of epochs, with some more control printouts.
"""

# A cycle is a set of training epochs and after that a set of test epochs.
# i.e. 1000 training epochs, 100 test epochs, repeat for 100 loops (cycles).
def cycle(trainingEpochs, testEpochs, totalCycles):
    #Create table and deck for training
    trainingTable = table.Table(16, 0.01, 0.9)
    trainingTable.maximumEpoch = trainingEpochs  # Set total epochs in table #TODO might be redundant now
    d = deck.Deck()

    # create separate table for testing (since it requires an AI team vs. a random team). Also requires player.testing to be set to 'True'. And, most importantly, needs the player objects from the trainingTable
    testingTable = table.Table(16, 0.01, 0.9)

    for i in range(totalCycles):
        print('Updating the training table')
        trainingTable = updateTrainingTable(trainingTable)
        print('Training...')
        training(trainingTable, d, trainingEpochs)
        print('Updating the test table')
        testingTable = updateTestingTable(trainingTable, testingTable)
        print('Testing...')
        training(testingTable, d, testEpochs)
        print("Cycle " +str(i+1)+ " out of " + str(totalCycles) + " finished\n")

        
    printResults(testingTable)

def training(t, d, trainingEpochs):

    #load parameters here
    #if loadP == 0:
    #    t.LoadState(SAVEFILELIST)
    #elif loadP == 0:
    #    print("Model parameters not loaded")

    start = time.time()

    for currentEpoch in range(trainingEpochs):
        if t.players[0].testing == True: input()
        #Set experiment information in table
        t.currentEpoch = currentEpoch

        d.SetTrump(rnd.choice(d.suits))
        d.DivideCards()
        print(d.dividedCards)
        t.DealCards(d)
        print('hands', t.players[0].handAsTuple())

        while t.players[0].hand != []:
            t.PlayCards(d)
            winner = t.WhoWinsTrick(d)
            print(t.players[0].testing)
            if t.players[0].testing == False:
                t.DoBackprop()
            elif t.players[0].testing == True:
                print('here')
                print(t.testingScore)
                t.testingScore.append(t.roundScore.copy())

                    

        if currentEpoch % printEpoch == 0: print("Epoch {} of {} \t\t\tElapsed time: {:.4} s".format(currentEpoch, trainingEpochs, time.time() - start))
    #    if currentEpoch == 100:
    #        t.SaveState(SAVEFOLDER)
    #        print("Saved model parameters")


#print("Training completed succesfully! \tElapsed time: {:.4} s \nShowing plots...".format(time.time() - start))

def updateTestingTable(trainingTable, testingTable):
    # Copy the networks for the first team to the testingTable ## copy the player objects directly
    for i in (0,2):
        testingTable.players[i] = trainingTable.players[i]
        testingTable.SetPlayerBehaviour(i,'Network')       #just to be sure, they should be 'Network' by default        

    # Set second team to random AI
    for i in (1,3):
        testingTable.SetPlayerBehaviour(i, 'Random')
        
    # Set boolean 'testing' to True for all players
    for p in testingTable.players:
        p.testing = True
       
    return testingTable

def updateTrainingTable(trainingTable):      #needed to reset the players.testing boolean
    for p in trainingTable.players:
        p.testing = False

    return trainingTable

def testing(updatedTestingTable, d, testingEpochs):

    training(updatedTestingTable, d, testingEpochs) # Reusing old code with new tables and after setting players.testing == True

def printResults(t):
    graphs  = [np.array(t.testingScores[i]) for i in (0,1)]    #the testing scores for the teams, where team 0 is network playing and team 1 is random
   # wGraphs = [np.array(p.weightedRewardArray) for p in t.players]
    styles = ['-b', '-r']
   # plt.subplot(121)
    p =[plt.plot(np.cumsum(graphs[i]), s, label='Team '+str(i)) for i,s in enumerate(styles)]
    plt.legend()
    plt.title('Team scores during testing')
    plt.xlabel('Testing tricks')
    plt.ylabel('Team scores')
    plt.grid()
   # plt.subplot(122)
   # pp =[plt.plot(np.cumsum(wGraphs[i]), s, label='Player '+str(i)) for i,s in enumerate(styles)]
   # plt.legend()
   # plt.title('Weighted reward')
   # plt.xlabel('Tricks')
   # plt.ylabel('Reward / hand value')
   # plt.grid()
    plt.show()

#print("Program terminated! \t\tTotal running time: {:.5} s".format(time.time() - start))



# The interesting part:
cycle(100, 100, 2)


