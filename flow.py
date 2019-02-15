import os
import matplotlib.pyplot as plt
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


printEpoch, saveEpoch = 10000, 1000
start = time.time()
COMMENT_STRING = ""

def cycle(trainingEpochs, testEpochs, totalCycles, learningRate, discountFactor):
    #Create table and deck for training
    trainingTable = table.Table(16, learningRate, discountFactor)
    # Set total epochs in table
    trainingTable.maximumEpoch = trainingEpochs * totalCycles
    d = deck.Deck()
    # create separate table for testing
    testingTable = table.Table(16, learningRate, discountFactor)
    # First test for baseline
    testingTable = updateTestingTable(trainingTable, testingTable)
    testing(testingTable, d, testEpochs)
    for currentCycle in range(totalCycles):
        print('Updating the training table')
        trainingTable = updateTrainingTable(trainingTable)
        print('Training...')
        training(trainingTable, d, trainingEpochs)
        print('Updating the test table')
        testingTable = updateTestingTable(trainingTable, testingTable)
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
        while t.players[0].hand != []:
            sim.run(d, t)
            t.PlayCards(d)
            winner = t.WhoWinsTrick(d)
            t.testingScores.append(t.roundScore.copy())
    t.SetPlayerBehaviour(0, "Network") #Back to the original values, as sim.run changes them.
    t.SetPlayerBehaviour(2, "Network")

def training(t, d, trainingEpochs):
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

def updateTestingTable(trainingTable, testingTable):
    # Copy the networks for the first team to the testingTable
    for i in (0,2):
        testingTable.players[i].net.load_state_dict(trainingTable.players[i].net.state_dict())    
        testingTable.SetPlayerBehaviour(i,'Network')      
    # Set second team to random AI
    for i in (1,3):
        testingTable.SetPlayerBehaviour(i, 'Random')        
    for p in testingTable.players:
        p.testing = True
    return testingTable

def updateTrainingTable(trainingTable):
    for p in trainingTable.players:
        p.testing = False
    for i in range(4):
        trainingTable.SetPlayerBehaviour(i, 'Network')
    #Set opponents to the previous iteration of the network, but don't train this network.
    if SELFPLAY_OPPONENTS:
        trainingTable.players[1].net.load_state_dict(trainingTable.players[0].net.state_dict())  
        trainingTable.players[3].net.load_state_dict(trainingTable.players[2].net.state_dict())
        trainingTable.players[1].testing = True 
        trainingTable.players[3].testing = True
    return trainingTable

def testing(updatedTestingTable, d, testingEpochs):
    training(updatedTestingTable, d, testingEpochs)
    updatedTestingTable.calculateTestResults()
    print('Testing completed')
    print('>>>Winrate (this cycle/mean/max): ' + str(updatedTestingTable.testingWinRatioTeam0[-1]) + " / " +
                                                   str(np.mean(updatedTestingTable.testingWinRatioTeam0)) +
                                                   " / " + str(np.max(updatedTestingTable.testingWinRatioTeam0))
          + ' ; Score (this cycle/mean/max): ' + str(updatedTestingTable.testingCycleScoresTeam0[-1]) + " / " +
                                                str(np.mean(updatedTestingTable.testingCycleScoresTeam0)) +
                                             " / " + str(np.max(updatedTestingTable.testingCycleScoresTeam0)))
   
def printResults(t, trainingEpochs, testEpochs, totalCycles):
    epochString = "\n Totals: Traningepochs=" + str(totalCycles * trainingEpochs) + " Testingepochs=" + str(
        totalCycles * testEpochs)
    currentTime = time.strftime("%Y%m%d-%H%M%S")
    # the testing scores for the teams, where team 0 is network playing and team 1 is random
    plotDataScores = [[], []]
    for i in range(len(t.testingCycleScoresTeam0)):
        plotDataScores[0].append(t.testingCycleScoresTeam0[i])
        plotDataScores[1].append(t.testingCycleScoresTeam1[i])
    plt.subplot(121)
    plt.plot((plotDataScores[0]), '-b', label='Team 0 - network play')
    plt.legend()
    plt.title('Average final round scores') 
    plt.xlabel('Testing cycle')
    plt.ylabel('Team scores')
    plt.grid()
    plt.subplot(122)
    plotDataWinratios = [[], []]
    for i in range(len(t.testingCycleScoresTeam0)):
        plotDataWinratios[0].append(t.testingWinRatioTeam0[i])
        plotDataWinratios[1].append(1-t.testingWinRatioTeam0[i])
    plt.plot((plotDataWinratios[0]), '-b', label='Team 0 - network play')
    plt.legend()
    plt.title('Winrate')
    plt.xlabel('Testing cycle')
    plt.ylabel('Winrate ratio')
    plt.grid()
    saveToFile(t, epochString, currentTime, plotDataScores, plotDataWinratios)
    plt.savefig(os.path.dirname(__file__) + '/data/' + currentTime + '/figure1.png')
    plt.close()
    plt.plot((t.testingTeam0TotalWins), '-b', label='Team 0 - network play')
    plt.plot((t.testingTeam1TotalWins), '-r', label='Team 1 - random play')
    plt.legend()
    plt.title('Won games' + epochString)
    plt.xlabel('Testing games played')
    plt.ylabel('Games won')
    plt.grid()
    #plt.show()    #uncomment to show plot directly
    plt.savefig(os.path.dirname(__file__) + '/data/' + currentTime + '/figure2.png')   #uncomment to save plots to file

    saveToFile(t, epochString, currentTime, plotDataScores, plotDataWinratios)
    print("Program terminated! \t\tTotal running time: {:.5} minutes".format((time.time() - start) /60))

def saveToFile(table, epochString, currentTime, scores, winrateRatio):
    SAVEFOLDER = os.path.dirname(__file__) + '/data/' + currentTime
    if not os.path.exists(SAVEFOLDER):
        os.makedirs(SAVEFOLDER)
    data = np.concatenate((np.array(scores), np.array(winrateRatio)))
    opponentTypeString = ""
    if SELFPLAY_OPPONENTS:
        opponentTypeString = "Previous iteration network as training opponent"
    headerString = "Data in order of lines: Average scores team 0 for each cycle, Score team 1, winrate team 0, winrate team 1\nHyperparameters: learningrate = " + str(table.players[0].alpha)+ "; discountrate = " + str(table.players[0].y) + "; explorationrate = " + str(table.players[0].epsilon) + "Opponent type: " + opponentTypeString + epochString + "\nNetwork: " + str(table.players[0].net) + "\nComments:" + str(COMMENT_STRING)
    np.savetxt(SAVEFOLDER + "/data.csv", data, fmt='%2f', delimiter=",",  header=headerString)


SELFPLAY_OPPONENTS = True
cycle(10000, 100, 100, 0.0001, 0.9)
