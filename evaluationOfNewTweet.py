from datetime import date
from tkinter import messagebox
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
#this function takes as input text, and the userlocation and return the prediction value
def evaluationOfNewTweet(tweetText,userLoc):
    randomVariables = {}
    analyzer = SentimentIntensityAnalyzer()

    # read the file that contains the trained model
    with open('models/model.txt') as f:
        content = f.readlines()
        content = [x.strip() for x in content]

    trainedModelStr = ""
    # create model based on txt file
    for i in range(1, len(content)):
        trainedModelStr = trainedModelStr + content[i] + "\n"

    for i in range(1, len(content)):
        if ":-" in content[i]:
            break
        strSplit = content[i].split("::")
        randomVar = strSplit[1].split(".")
        randomVariables[randomVar[0]] = False

    evidence = []
    evidenceDict = {}
    #this proccess is for evaluation of one tweet at the time
    evidenceDict = randomVariables.copy()
    p = PrologString(trainedModelStr)
    engine = DefaultEngine()
    db = engine.prepare(p)
    lf = engine.ground_all(db)
    query = Term('visitLocation')
    today = date.today()
    tweet1 = NewTweet(tweetText,today , userLoc)
    evidenceDict = sentiment_analyzer_scores(tweet1.text, analyzer,
                                             evidenceDict)
    evidenceDict = readRelatedWordsDict(tweet1.text,
                                        evidenceDict)
    evidenceDict = checkPlace(tweet1.location,
                              evidenceDict)
    evidenceDict = {Term(k): v for k, v in evidenceDict.items()}
    evidence = [(key, value) for key, value in evidenceDict.items()]
    lf = engine.ground_all(db, evidence=evidence, queries=[query])
    result = get_evaluatable().create_from(lf).evaluate()

    for tweet, score in result.items():
        if score > 0.50:
           return score
        else:
            return score

