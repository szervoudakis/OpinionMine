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
#the createModel.py contains all the training process
#the array examples.npy contains all evidences based on that create the trained model (model.txt)
#in array oldEvidence is all the evidences (from example.npy)
#this function is for incremental learning proccess
#currentEvidence is the array that contains all evidences based on new train set
#in array newEvidence that has all evidences (from model.txt model , and the current model)
def incrementalLearning(currentRandomVars, sizeOfCurrentSet, currentEvidence):
    oldRandomVariables = {}
    newRandomVariables = {}
    evidenceOld = {}
    newEvidence = []
    txtEvidence = {}
    content = []
    #if model exist open and read the content
    if os.path.exists('models/model.txt'):
        with open('models/model.txt') as f:
            content = f.readlines()
            content = [x.strip() for x in content]

    if os.path.exists('models/examples.npy'):
        numpyArray = np.load("models/examples.npy")
        oldEvidence = []
        oldEvidence = convertToSet(numpyArray)
        newEvidence = oldEvidence + currentEvidence
    else:
        newEvidence = currentEvidence

    if len(content) > 0:
        total = int(content[0]) + sizeOfCurrentSet
        percentageOfCurrentDataset = (100 * sizeOfCurrentSet) / total
        percentageOfOldDataset = 100 - percentageOfCurrentDataset
        for i in range(1, len(content)):
            if ":-" in content[i]:
                break
            variable = content[i].split("::")
            tempVar = variable[1].split(".")
            oldRandomVariables[tempVar[0]] = variable[0]
    else:
        percentageOfCurrentDataset = 100

    facts = ""
    #if our model has evidences create the updated probabilities from random variables
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
    return newEvidence, facts, len(newEvidence) #this function return newevidence(the new array from all evidences, from old and current model), facts, and the len of evidences
#this function takes as input the file name (csv file) and returns the train model
def createModel(fileName):
    tweetsList = readCSV(fileName)
    analyzer = SentimentIntensityAnalyzer()
    # in variable facts create all the random variables
    facts = """t(_)::userLocation.\n"""
    neg = False
    pos = False
    keywordsList = ['']
    examples = []
    randomVariables = {}
    # create random variables based on dataset
    for i in range(1, len(tweetsList)):
        # if exists in csv negative sentiment create the random var negativeSentiment
        if sentiment_analyzer(tweetsList[i].text, analyzer) == -1 and neg == False:
            # print(i)
            facts = facts + "t(_)::negativeSentiment.\n"
            neg = True
        # if exists in csv positive sentiment create the random var negativeSentiment positiveSentiment
        if sentiment_analyzer(tweetsList[i].text, analyzer) == 1 and pos == False:
            facts = facts + "t(_)::positiveSentiment.\n"
            pos = True

        keywordsList.append(readRelatedWords(tweetsList[i].text))

    keywordsList = list(dict.fromkeys(keywordsList))  #delete all duplicates
    keywordsList.remove('')
    temp = []

    for i in range(0, len(keywordsList)):
        facts = "" + facts + "t(_)::" + keywordsList[i] + ".\n"

    # create the random variables in problog syntax
    for line in facts.splitlines():
        if ":-" not in line:
            temp = line.split("t(_)::")
            variable = temp[1].split(".")
            randomVariables[variable[0]] = False

    for i in range(0, len(keywordsList)):
        randomVariables[keywordsList[i]] = False

    initialDict = randomVariables.copy()
    # with these for loop in list example put all evidences
    for i in range(1, len(tweetsList)):
        tempDict = initialDict.copy()
        # print(i)
        tempDict = sentiment_analyzer_scores(tweetsList[i].text, analyzer,
                                             tempDict)  # sentiment analysis in each tweet
        tempDict = readRelatedWordsDict(tweetsList[i].text,
                                        tempDict)  # NER in tweets
        tempDict = checkPlace(tweetsList[i].location,
                              tempDict)  #

        orderedDictionary = collections.OrderedDict(sorted(tempDict.items()))
        orderedDictionary = {Term(k): v for k, v in
                             orderedDictionary.items()}  # convert each key from dictionary to Term
        examples.append([(key, value) for key, value in
                         orderedDictionary.items()])  # append the instance of evidence in examples


    c = Counter(tuple(x) for x in iter(examples))
    newString = ""
    newExamples = []

    # here create the rules based on evidence set
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

    # begin train of model based on evidences and the current model
    score, weights, atoms, iteration, lfi_problem = lfi.run_lfi(PrologString(finalModel), examples)
    trainedModel = lfi_problem.get_model()

    tempForSplit = trainedModel.split("\n")
    currentRandomVars = {}

    for i in range(0, len(tempForSplit)):
        if ":-" in tempForSplit[i]:
            break
        pos = tempForSplit[i].split('::')
        key = pos[1].split(".")
        currentRandomVars[key[0]] = pos[0]

    # begin incremental learning
    newExamples, facts, lenOfSet = incrementalLearning(currentRandomVars, len(tweetsList), examples)

    rulesAndNum = {}
    convertArr = []
    for i in range(0,len(newExamples)):
        convertArr.append([(str(k),v) for (k,v) in newExamples[i]])

    arrayForNewEvidence=[]
    for i in range(0,len(newExamples)):
      arrayForNewEvidence.append([(str(k),v) for k,v in newExamples[i] if (v==True)])

    rulesAndNum = Counter(tuple(y) for (y) in iter(arrayForNewEvidence))
    #create the rules based on evidence set
    finalRules = ""
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

    # store the trained model in txt file
    if os.path.exists('models/model.txt'):
        os.remove('models/model.txt')

    text_file = open("models/model.txt", "w")
    text_file.write(str(lenOfSet) + "\n")
    text_file.write(finalModelIncremental)
    text_file.close()
    print('the model trained successfully')
