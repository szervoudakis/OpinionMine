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

def eval(file):
    randomVariables = {}
    y_pred=[]
    analyzer = SentimentIntensityAnalyzer()
    # diavazw to text arxeio opou einai oi trained rules kai ta facts
    with open('models/model.txt') as f:
        content = f.readlines()
        content = [x.strip() for x in content]

    tweetsList = []
    # εδω διαβαζω το testSet και αποθηκευω τα δεδομενα του σε ενα πινακα
    with open('datasets/' + file, 'r', encoding='utf-8-sig') as csvFile:
        reader = csv.reader(csvFile)
        # εδω η επαναληπτικη εντολη ουσιαστικα παιρνει γραμμη - γραμμη τα στοιχεια που ειναι αποθηκευμενα
        # στο CSV αρχειο και τα τοποθετει
        # στην λιστα. Το εκανα αυτο διοτι, μια λιστα μπορω να την διαχειριστω πιο ευκολα
        # απο οτι ενα αρχειο
        for row in reader:
            temp = collectWords(row[0])
            hashtags = collectHashTags(temp)
            tweetsList.append(
                Tweet(row[0], row[1], row[2], row[3],
                      hashtags))  # στην λιστα κανω append καθε στιγμιοτυπο, το καθε στιγμιοτυπο περιεχει
            # τα εξης, το κειμενο, το username, την τοποθεσια του χρηστη και τα hashtags που χρησιμοποιησε
            # στο tweet που εκανε
    csvFile.close()

    trainedModelStr = ""
    # create model based on txt
    for i in range(1, len(content)):
        trainedModelStr = trainedModelStr + content[i] + "\n"

    # edw dhmiourgw to dictionary me tis tuxaies metavlhtes pou uparxoun sto montelo
    for i in range(1, len(content)):
        if ":-" in content[i]:
            break
        strSplit = content[i].split("::")
        randomVar = strSplit[1].split(".")
        randomVariables[randomVar[0]] = False

    # print(trainedModelStr)
    p = PrologString(trainedModelStr)
    engine = DefaultEngine()
    db = engine.prepare(p)
    lf = engine.ground_all(db)
    query = Term('visitLocation')
    print(db)
    count = 0
    TruePositive = 0
    for i in range(1, len(tweetsList)):
        evidence = []
        evidenceDict = {}
        evidenceDict = randomVariables.copy()
        print(i, '----------------------------------new tweet------------------------------------------------------')
        tweet1 = NewTweet(tweetsList[i].text, tweetsList[i].create, tweetsList[i].location)
        evidenceDict = sentiment_analyzer_scores(tweet1.text, analyzer,
                                                 evidenceDict)  # arxika tsekaroume ean einai 8etikh h arnhtikh h protash
        evidenceDict = readRelatedWordsDict(tweet1.text,
                                            evidenceDict)  # edw elegxoume ean uparxoun kapoia entities wste na ginoun oi antistoixes metavlhtes true
        evidenceDict = checkPlace(tweet1.location,
                                  evidenceDict)  # edw elegxoume ean h topo8esia apo tin opoia katagete o xrhsths einai ektos krhths
        evidenceDict = {Term(k): v for k, v in evidenceDict.items()}
        evidence = [(key, value) for key, value in evidenceDict.items()]
        lf = engine.ground_all(db, evidence=evidence, queries=[query])
        result = get_evaluatable().create_from(lf).evaluate()
        count = count + 1
        for tweet, score in result.items():
            TruePositive = TruePositive + score
            print(score);
            y_pred.append(score)

    evidence = []
    evidenceDict = {}
    evidenceDict = randomVariables.copy()
    print(count)
    print("-----------------------------------------------------------------------")
    print("This dataset contains ", count, " tweets")
    print("truepos",TruePositive)
    finalPososto = TruePositive / count
    # print("to pososto episkepshs ths Krhths einai ", finalPososto, "")


    return y_pred
# tweet1 = NewTweet(textTest,'2/5/19', 'Paris')
# evidenceDict = sentiment_analyzer_scores(tweet1.text, analyzer,
#                                          evidenceDict)  # arxika tsekaroume ean einai 8etikh h arnhtikh h protash
# evidenceDict = readRelatedWordsDict(tweet1.text,
#                                     evidenceDict)  # edw elegxoume ean uparxoun kapoia entities wste na ginoun oi antistoixes metavlhtes true
# evidenceDict = checkPlace(tweet1.location,
#                           evidenceDict)  # edw elegxoume ean h topo8esia apo tin opoia katagete o xrhsths einai ektos krhths
# evidenceDict = {Term(k): v for k, v in evidenceDict.items()}
# evidence = [(key, value) for key, value in evidenceDict.items()]
# lf = engine.ground_all(db, evidence=evidence, queries=[query])
# result = get_evaluatable().create_from(lf).evaluate()
# print(evidenceDict)
#test1=eval('testSet/trump.csv')
# test2=eval('testSet/test2_20.csv')
# test3=eval('testSet/test3_20.csv')
y_pred=[]
y_pred=eval('testSet/testSet.csv')
y_true=[]
#edw vazoume tin timh alh8eias, pou einai h megisth pi8anotita pou pairnei to montelo gia thn episkepsh sthn krhth
for i in range(len(y_pred)):
  y_true.append(1.00)

print("The mean squared error of our system is (MSE): ",mean_squared_error(y_true, y_pred))
print("The root mean squared error of our system is (RMSE): ",sqrt(mean_squared_error(y_true, y_pred)))
print("The mean absolute error of our system is (MAE):",mean_absolute_error(y_true, y_pred))
#
# generate random data-set
#np.random.seed(0)
x = y_true
# y_true=[1,1,1]
# y_pred=[0.4,0.3,0.1]

with open('metrics.csv', 'w', newline='') as csvfile:
    fieldnames = ['actual', 'predicted_value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(1,len(y_pred)):
        temp = str(y_pred[i])
        writer.writerow({'actual': '1', 'predicted_value': temp})

