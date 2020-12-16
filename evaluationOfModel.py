from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

from problog.program import PrologString
from problog.core import ProbLog
from problog import get_evaluatable
from problog.engine import DefaultEngine
from problog.logic import Term
from problog.learning import lfi
from problog.tasks import sample
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from EntitiesOfTweet import NewTweet
from functions import sentiment_analyzer_scores, readRelatedWordsDict, checkPlace
from functions import *
import sys
import csv
import pandas as pd
import collections
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

#this function is for evaluation of model, this function takes as input the csv file and returns the metrics of RMSE, MAE,MSE
def evaluationOfTrainedModel(file):
    randomVariables = {}
    y_pred=[]
    analyzer = SentimentIntensityAnalyzer()
    #read the model.txt if exists, and store the content in list content
    with open('models/model.txt') as f:
        content = f.readlines()
        content = [x.strip() for x in content]

    tweetsList = []

    with open('datasets/' + file, 'r', encoding='utf-8-sig') as csvFile:
        reader = csv.reader(csvFile)
        # read csv for tweets row by row and store them in list
        for row in reader:
            temp = collectWords(row[0])
            hashtags = collectHashTags(temp)
            tweetsList.append(
                Tweet(row[0], row[1], row[2], row[3],
                      hashtags))   #put the instance of Tweet class in list
    csvFile.close() #close the csv file

    trainedModelStr = ""
    # create model based on txt
    for i in range(1, len(content)):
        trainedModelStr = trainedModelStr + content[i] + "\n"

    # in this for loop takes all random variables in dictionary
    for i in range(1, len(content)):
        if ":-" in content[i]:
            break
        strSplit = content[i].split("::")
        randomVar = strSplit[1].split(".")
        randomVariables[randomVar[0]] = False

    p = PrologString(trainedModelStr)
    engine = DefaultEngine()
    db = engine.prepare(p)
    lf = engine.ground_all(db)
    query = Term('visitLocation')
    count = 0
    TruePositive = 0
    for i in range(1, len(tweetsList)):
        evidence = []
        evidenceDict = {}
        evidenceDict = randomVariables.copy()
        # print(i, '----------------------------------new tweet------------------------------------------------------')
        tweet1 = NewTweet(tweetsList[i].text, tweetsList[i].create, tweetsList[i].location)
        evidenceDict = sentiment_analyzer_scores(tweet1.text, analyzer,
                                                 evidenceDict)  # sentiment analysis, check if sentiment is neg or positive
        evidenceDict = readRelatedWordsDict(tweet1.text,
                                            evidenceDict)  #NER in tweets
        evidenceDict = checkPlace(tweet1.location,
                                  evidenceDict)  # check if user is from ccrete
        evidenceDict = {Term(k): v for k, v in evidenceDict.items()}
        evidence = [(key, value) for key, value in evidenceDict.items()]
        lf = engine.ground_all(db, evidence=evidence, queries=[query])
        result = get_evaluatable().create_from(lf).evaluate()
        count = count + 1
        for tweet, score in result.items():
            y_pred.append(score)

    evidence = []
    evidenceDict = {}
    evidenceDict = randomVariables.copy()

    finalPososto = TruePositive / count

    return y_pred
