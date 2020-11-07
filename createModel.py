import ast
import json
import os
from tempfile import TemporaryFile

import numpy
import problog
import time
from problog.cnf_formula import CNF
from problog.ddnnf_formula import DDNNF
from problog.formula import LogicDAG
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from EntitiesOfTweet import *
from functions import *
from problog.program import PrologString
from problog.core import ProbLog
from problog import get_evaluatable
from problog.engine import DefaultEngine
from problog.logic import Term
from problog.learning import lfi
from problog.tasks import sample
from problog.tasks.dtproblog import dtproblog
import collections
from collections import Counter
from itertools import chain
import sys
import csv
import pandas as pd
import numpy as np

def incrementalLearning(currentRandomVars, sizeOfCurrentSet, currentEvidence):
    oldRandomVariables = {}
    newRandomVariables = {}
    evidenceOld = {}
    newEvidence = []
    txtEvidence = {}
    content = []

    if os.path.exists('models/model.txt'):
        with open('models/model.txt') as f:
            content = f.readlines()
            content = [x.strip() for x in content]

    # ean exei ginei ma8hsh me kapoio dataset
    if os.path.exists('models/examples.npy'):
        # diavazw to array pou exw apo8hkeuvmeno sto numpy , to kanw convert se kanoniko set opws to examples
        numpyArray = np.load("models/examples.npy")
        oldEvidence = []
        oldEvidence = convertToSet(numpyArray)
        newEvidence = oldEvidence + currentEvidence
    else:
        newEvidence = currentEvidence

    # edw vgazw to pososto pou
    # 8a parei to ka8e dataset, an uparxei palio
    if len(content) > 0:
        total = int(content[0]) + sizeOfCurrentSet
        percentageOfCurrentDataset = (100 * sizeOfCurrentSet) / total
        percentageOfOldDataset = 100 - percentageOfCurrentDataset
        print("palio einai toso ", percentageOfOldDataset)
        print("current einai toso ", percentageOfCurrentDataset)
        # pairnw apo to palio model
        # tis metavlites kai ta varh tous
        for i in range(1, len(content)):
            if ":-" in content[i]:
                break
            variable = content[i].split("::")
            tempVar = variable[1].split(".")
            oldRandomVariables[tempVar[0]] = variable[0]
    else:
        percentageOfCurrentDataset = 100

    facts = ""
    # an uparxei palio dataset
    if os.path.exists('models/examples.npy'):
        for key, value in currentRandomVars.items():
            if checkKey(oldRandomVariables, key) == 1:
                sumWeight = float((percentageOfOldDataset / 100)) * float(oldRandomVariables[key])
                sumWeight = sumWeight + float((percentageOfCurrentDataset / 100)) * float(value)
                newRandomVariables[key] = sumWeight
            else:
                sumWeight1 = float(value)*100/total
                newRandomVariables[key] = sumWeight1

        for key,value in oldRandomVariables.items():
            if checkKey(currentRandomVars,key)==0:
                sumWeight2= float(value)*100/total
                newRandomVariables[key]=sumWeight2
        for key, value in newRandomVariables.items():
            facts = facts + str(value) + "::" + key + ".\n"
    else:
        for key, value in currentRandomVars.items():
            facts = facts + str(value) + "::" + key + ".\n"

    np.save("models/examples.npy", np.array(newEvidence))
    return newEvidence, facts, len(newEvidence)


def createModel(fileName):
    tweetsList = readCSV(fileName)
    analyzer = SentimentIntensityAnalyzer()
    # στην μεταβλητη facts δημιουργουνται ολες οι τυχαιες μεταβλητες
    facts = """t(_)::userLocation.\n"""
    neg = False
    pos = False
    keywordsList = ['']
    examples = []
    randomVariables = {}  # einai to dictionary opou apo8hkeuontai oi tuxaies metablites pou 8a dhmiourgh8oun parakatw
    # create random variables based on dataset
    for i in range(1, len(tweetsList)):
        # εαν στην λιστα που ειναι αποθηκευμενα τα tweets υπαρχουν αρνητικα tweets τοτε δημιουργησε την τυχαια μεταβλητη negativeSentiment
        if sentiment_analyzer(tweetsList[i].text, analyzer) == -1 and neg == False:
            print(i)
            facts = facts + "t(_)::negativeSentiment.\n"
            neg = True
        # εαν στην λιστα που ειναι αποθηκευμενα τα tweets υπαρχουν θετικα tweets τοτε δημιουργησε την τυχαια μεταβλητη positiveSentiment
        if sentiment_analyzer(tweetsList[i].text, analyzer) == 1 and pos == False:
            facts = facts + "t(_)::positiveSentiment.\n"
            pos = True
        # εδω καλουμε την συναρτηση readRelatedWord στην οποια ελεγχουμε τι υπαρχει απο τις τυχαιες μεταβλητές-κατηγοριες στο tweet
        keywordsList.append(readRelatedWords(tweetsList[i].text))

    print(keywordsList)
    keywordsList = list(dict.fromkeys(keywordsList))  # afairw ta dipla wste na uparxei mia fora h ka8e tuxaia metablhth
    keywordsList.remove('')
    print(keywordsList)
    temp = []
    # dhmiourgw tis metavlites pou sxetizontai me to keywords variable
    for i in range(0, len(keywordsList)):
        facts = "" + facts + "t(_)::" + keywordsList[i] + ".\n"

    # edw dhmiourgw tis metavlhtes me vash ta facts pou exoun dhmiourgh8ei parapanw
    for line in facts.splitlines():
        if ":-" not in line:
            temp = line.split("t(_)::")
            variable = temp[1].split(".")
            randomVariables[variable[0]] = False

    for i in range(0, len(keywordsList)):
        randomVariables[keywordsList[i]] = False
    # edw dhmiourgoume tis metavlhtes pou 8a exoun ton tin idiotita Term.

    # edw 8a dhmiourghsw to evidence set, pou me vash autou, 8a dhmiourgh8oun oi kanones dunamika
    initialDict = randomVariables.copy()
    examples1 = []
    numpyExamples = []
    testar = []
    # σε αυτη την επαναληπτικη διαδικασια, ουσιαστικα βάζει στην lista examples ολα τα evidence
    for i in range(1, len(tweetsList)):
        tempDict = initialDict.copy()
        # print(i)
        tempDict = sentiment_analyzer_scores(tweetsList[i].text, analyzer,
                                             tempDict)  # edw pairnei times gia sentiment analysis 0 -1 1
        tempDict = readRelatedWordsDict(tweetsList[i].text,
                                        tempDict)  # edw pairnei true false times gia tis metavlhtes keywords pou dhmiourgh8hkan
        tempDict = checkPlace(tweetsList[i].location,
                              tempDict)  # edw tsekarei to user location tou xrhsth pou ekane to tweet (ean einai apo tin krhth h oxi)

        orderedDictionary = collections.OrderedDict(sorted(tempDict.items()))
        orderedDictionary = {Term(k): v for k, v in
                             orderedDictionary.items()}  # edw to key tou dictionary to kanei convert se Term
        examples.append([(key, value) for key, value in
                         orderedDictionary.items()])  # edw kanei append ta evidence pou einai apo8hkeumena sto tempDict

    # το c ειναι μια λιστα η οποια περιεχει evidences και ποσες φορες εμφανιστικαν το καθενα
    c = Counter(tuple(x) for x in iter(examples))
    newString = ""
    newExamples = []

    # eprepe na kanw install tin 16.2 version gia na dior8w8ei to error allow_pickle false
    print(np.__version__)

    # εδω δημιουργουνται οι κανονες
    for key, value in c.most_common():
        rules = "visitLocation:-"
        final = value / len(tweetsList)
        probability = "t(" + str(final) + ")::"
        rules = probability + rules
        for (i, j) in key:
            if j != None:
                if j == False:
                    pass
                else:
                    rules = rules + "" + str(i) + ","
        newString += rules[:-1]
        newString = newString + ".\n"
    finalModel = facts + newString

    # στην συναρτηση αυτη, δινουμε ως ορισμα, το μοντελο (δλδ τα facts και τα τους κανονες που δημιουργηθηκαν με βαση τα evidence)
    score, weights, atoms, iteration, lfi_problem = lfi.run_lfi(PrologString(finalModel), examples)
    trainedModel = lfi_problem.get_model()

    tempForSplit = trainedModel.split("\n")
    currentRandomVars = {}
    # apo to ekpedeumeno montelo 8elw ta varh twn tuxaiwn metavlitwn, wste na ta xrhsimopoihsw sto incremental algori8mo pou eftiaksa
    for i in range(0, len(tempForSplit)):
        if ":-" in tempForSplit[i]:
            break
        pososto = tempForSplit[i].split('::')
        key = pososto[1].split(".")
        currentRandomVars[key[0]] = pososto[0]

    # # edw kalw ton algori8mo pou kanei to incremental learning
    newExamples, facts, lenOfSet = incrementalLearning(currentRandomVars, len(tweetsList), examples)
    # το c1 ειναι μια λιστα η οποια περιεχει evidences και ποσες φορες εμφανιστικαν το καθενα
    rulesAndNum = {}
    convertArr = []
    for i in range(0,len(newExamples)):
        convertArr.append([(str(k),v) for (k,v) in newExamples[i]])

    kialo=[]
    for i in range(0,len(newExamples)):
      kialo.append([(str(k),v) for k,v in newExamples[i] if (v==True)])

    rulesAndNum = Counter(tuple(y) for (y) in iter(kialo))
    # εδω δημιουργουνται οι κανονες
    finalRules = ""
    print(facts)
    for key, value in rulesAndNum.most_common():
        rules = "visitLocation:-"
        final = value / (lenOfSet - 1)
        probability = "" + str(final) + "::"
        rules = probability + rules
        for (i, j) in key:
            if j != None:
                if j == False:
                    pass
                else:
                    rules = rules + "" + str(i) + ","
        finalRules += rules[:-1]
        finalRules = finalRules + ".\n"
    finalModelIncremental = facts + finalRules

    # αποθηκευω το εκπευδευμενο μοντελο σε ενα text
    # αρχειο, που οταν κανω το evaluation του μοντελου,
    # να διαβαζω το txt αρχειο
    # για να μην ξανατρεχει αυτο το σκριπτ παλι.
    if os.path.exists('models/model.txt'):
        os.remove('models/model.txt')
    # apo8hkeuw to trained model se txt arxeio
    text_file = open("models/model.txt", "w")
    text_file.write(str(lenOfSet) + "\n")
    text_file.write(finalModelIncremental)
    text_file.close()

#createModel('clear.csv')
print(np.__version__)
#numpy-1.17.4
