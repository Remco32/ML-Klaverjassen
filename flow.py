import table
import deck
import random as rnd
import learn
import montecarlo
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import time


#trainingEpochs, printEpoch, saveEpoch = 1000, 100, 10
printEpoch, saveEpoch = 1000, 1000
pauseTime = 0.1     # In seconds. For making sure lists and such are filled - filthy dirty workaround (aka a hack)

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

    #TODO: This is a workaround for a bug caused by untested code being pushed
    #training(trainingTable, d, 1)
    # First test for baseline
    testingTable = updateTestingTable(trainingTable, testingTable) #No need to use the training table since it hasn't trained yet
    testing(testingTable, d, testEpochs)

    for currentCycle in range(totalCycles):
        print('Updating the training table')
        trainingTable = updateTrainingTable(trainingTable)
        print('Training...')
        training(trainingTable, d, trainingEpochs)
        print('Updating the test table')
        testingTable = updateTestingTable(trainingTable, testingTable)
        #print('Running simulation')
        #simulate(testingTable, d, trainingEpochs)  # Uncomment to run, WARNING: TAKES A CONSIDERABLE AMOUNT OF TIME
        print('Testing...')
        testing(testingTable, d, testEpochs)
        print("Cycle " + str(currentCycle+1)+ " out of " + str(totalCycles) + " finished\n")
        remainingTime = (((time.time() - start) / 60) / (currentCycle+1)) * ((totalCycles+1)-(currentCycle+1))
        print("Expected time needed remaining cycles: {:.4} min".format(remainingTime))

    printResults(testingTable, trainingEpochs, testEpochs, totalCycles)

def simulate(t, d, trainingEpochs):
    ''' Runs the Monte Carlo simulation''' 
    for currentEpoch in range(trainingEpochs):
        #Set experiment information in table
        t.currentEpoch = currentEpoch
        sim = montecarlo.Simulation(d, t)
        d.SetTrump(rnd.choice(d.suits))
        d.DivideCards()
        t.DealCards(d)
        print("Simulating...")

        while t.players[0].hand != []:
            sim.run(d, t)
            t.PlayCards(d)
            winner = t.WhoWinsTrick(d)
            t.testingScores.append(t.roundScore.copy())
        if currentEpoch % printEpoch == 0:
            print("Simulation {} of {} \t\t\tElapsed time: {:.4} min".format(
                currentEpoch, trainingEpochs, (time.time() - start)/60))
    t.SetPlayerBehaviour(0, "Network")
    t.SetPlayerBehaviour(2, "Network")

def training(t, d, trainingEpochs):

    #load parameters here
    #if loadP == 0:
    #    t.LoadState(SAVEFILELIST)
    #elif loadP == 0:
    #    print("Model parameters not loaded")


    for currentEpoch in range(trainingEpochs):
        #Set experiment information in table
        t.currentEpoch = currentEpoch

        d.SetTrump(rnd.choice(d.suits))
        d.DivideCards()
        t.DealCards(d)

        while t.players[0].hand != []:
            t.PlayCards(d)
            winner = t.WhoWinsTrick(d)
            if t.players[0].testing == False:
                t.DoBackprop()
            elif t.players[0].testing == True:
                t.testingScores.append(t.roundScore.copy())

                    

        if currentEpoch % printEpoch == 0: print("Epoch {} of {} \t\t\tElapsed time: {:.4} min".format(currentEpoch, trainingEpochs, (time.time() - start)/60))
    #    if currentEpoch == 100:
    #        t.SaveState(SAVEFOLDER)
    #        print("Saved model parameters")

        #time.sleep(pauseTime) #Workaround for program accessing lists etc. that aren't filled yet after thousands of epochs, probably due to some CPU magic


#print("Training completed succesfully! \tElapsed time: {:.4} s \nShowing plots...".format(time.time() - start))

def updateTestingTable(trainingTable, testingTable):
    # Copy the networks for the first team to the testingTable ## copying the players directly won't work
    for i in (0,2):
        testingTable.players[i].net.load_state_dict(trainingTable.players[i].net.state_dict())     #trying to solve the problem by not copyin the players but the model
        testingTable.SetPlayerBehaviour(i,'Network')       #just to be sure, they should be 'Network' by default        

    # Set second team to random AI
    for i in (1,3):
        testingTable.SetPlayerBehaviour(i, 'Random')
        
    # Set boolean 'testing' to True for all players
    for p in testingTable.players:
        p.testing = True

    # Reset game score
       
    return testingTable

def updateTrainingTable(trainingTable):      #needed to reset the players.testing boolean
    for p in trainingTable.players:
        p.testing = False

    return trainingTable

def testing(updatedTestingTable, d, testingEpochs):

    training(updatedTestingTable, d, testingEpochs) # Reusing old code with new tables and after setting players.testing == True

    updatedTestingTable.calculateTestResults()
    print('Winrate team 0 this cycle: ' + str(updatedTestingTable.testingWinRatioTeam0[-1]))
    print('Testing completed')

    
def printResults(t, trainingEpochs, testEpochs, totalCycles):
    plotDataScores = [[], []]    #the testing scores for the teams, where team 0 is network playing and team 1 is random
    for i in range(len(t.testingCycleScoresTeam0)):
        plotDataScores[0].append(t.testingCycleScoresTeam0[i])
        plotDataScores[1].append(t.testingCycleScoresTeam1[i])

    #epochString = "\n Totals: Traningepochs=" + str(totalCycles*trainingEpochs) + " Testingepochs=" + str(totalCycles*testEpochs)
    epochString = ''
    plt.subplot(121)
    #plt.rcParams.update({'font.size': 12})

    plt.plot((plotDataScores[0]), '-b', label='Team 0 - network play')
    plt.plot((plotDataScores[1]), '-r', label='Team 1 - random play' )
    plt.legend()
    plt.title('Average round scores during testing' + epochString)
    plt.xlabel('Testing cycle')
    plt.ylabel('Team scores')
    plt.grid()
    #plt.show()


    plt.subplot(122)
    plotDataWinratios = [[], []]
    for i in range(len(t.testingCycleScoresTeam0)):
        plotDataWinratios[0].append(t.testingWinRatioTeam0[i])
        plotDataWinratios[1].append(1-t.testingWinRatioTeam0[i])


    plt.plot((plotDataWinratios[0]), '-b', label='Team 0 - network play')
    plt.plot((plotDataWinratios[1]), '-r', label='Team 1 - random play')
    plt.legend()
    plt.title('Winrate both teams' + epochString)
    plt.xlabel('Testing cycle')
    plt.ylabel('Winrate ratio')
    plt.grid()
    #plt.show()

    currentTime = time.strftime("%Y%m%d-%H%M%S")
    saveToFile(t, epochString, currentTime, plotDataScores, plotDataWinratios)
    plt.savefig(os.path.dirname(__file__) + '/data/' + currentTime + '/figure.png')

def saveToFile(table, epochString, currentTime, scores, winrateRatio):

    #Check for folder
    #currentTime = time.strftime("%Y%m%d-%H%M%S")

    SAVEFOLDER = os.path.dirname(__file__) + '/data/' + currentTime

    if not os.path.exists(SAVEFOLDER):
        os.makedirs(SAVEFOLDER)

    data = np.concatenate((np.array(scores), np.array(winrateRatio)))

    headerString = "Data in order of lines: Average scores team 0 for each cycle, Score team 1, winrate team 0, winrate team 1\nHyperparameters: learningrate = " + str(table.players[0].alpha)+ "; discountrate = " + str(table.players[0].y) + "; explorationrate = " + str(table.players[0].epsilon) + " Totals:" + epochString

    #TODO save dimensions (layers, amount of nodes) to file as well
    np.savetxt(SAVEFOLDER + "/data.csv", data, fmt='%2f', delimiter=",",  header=headerString)

#print("Program terminated! \t\tTotal running time: {:.5} s".format(time.time() - start))

start = time.time()
# The interesting part:
cycle(100, 100, 30)


# https://www.google.com/search?q=ValueError%3A+list.remove(x)%3A+x+not+in+list&oq=ValueError%3A+list.remove(x)%3A+x+not+in+list&aqs=chrome..69i57j69i58.286j0j1&sourceid=chrome&ie=UTF-8