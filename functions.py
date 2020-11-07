from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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
import sys
import csv
import pandas as pd

def readCSV(nameOfFile):
    tweetsList = []
    nameOfFile="datasets/"+nameOfFile
    # σε αυτο το script ουστιαστικα αποθηευω ολα τα tweets σε μια λιστα, η οποια περιεχει στιγμιοτυπα της κλασης Tweet (δειτε το αρχειο EntitiesOfTweet.py)
    with open(nameOfFile, 'r', encoding='utf-8-sig') as csvFile:
        reader = csv.reader(csvFile)
        # εδω η επαναληπτικη εντολη ουσιαστικα παιρνει γραμμη - γραμμη τα στοιχεια που ειναι αποθηκευμενα
        # στο CSV αρχειο και τα τοποθετει στην λιστα. Το εκανα αυτο διοτι, μια λιστα μπορω να την διαχειριστω πιο ευκολα
        # απο οτι ενα αρχειο
        count=0
        for row in reader:
            count=count+1
            temp = collectWords(row[0])
            hashtags = collectHashTags(temp)
            tweetsList.append(
                Tweet(row[0], row[1], row[2], row[3],
                      hashtags))  # στην λιστα κανω append καθε στιγμιοτυπο, το καθε στιγμιοτυπο περιεχει
            # τα εξης, το κειμενο, το username, την τοποθεσια του χρηστη και τα hashtags που χρησιμοποιησε
            # στο tweet που εκανε
    csvFile.close()

    return tweetsList

def sentiment_analyzer_scores(tweeterText, analyser, randomVariables):
    score = analyser.polarity_scores(tweeterText)
    lb = score['compound']
    if lb >= 0.05:
        randomVariables['positiveSentiment'] = True
        return randomVariables
    elif (lb > -0.05) and (lb < 0.05):
        randomVariables['positiveSentiment'] = True
        return randomVariables
    else:
        if checkKey(randomVariables,'negativeSentiment'):
         randomVariables['negativeSentiment'] = True
        else:
         randomVariables['positiveSentiment'] = None
    return randomVariables


def checkKey(dict, key):
    if key in dict.keys():
        return 1
    else:
        return 0

def sentiment_analyzer(tweetText, analyser):
    score = analyser.polarity_scores(tweetText)
    lb = score['compound']
    if lb >= 0.05:
        return 1
    elif (lb > -0.05) and (lb < 0.05):
        return 0
    else:
        return -1

def checkPlace(text,randomVariables):
    str = []
    with open('datasets/placesCopy.csv', 'r', encoding='utf-8-sig') as csvFile:
        reader = csv.reader(csvFile)
        str = text.split(",")
        # check if text contains some place of crete
        for row in reader:
            for i in range(len(str)):
                if row[0] == str[i].capitalize():
                    randomVariables['userLocation']=False
                    return randomVariables
                    csvFile.close()
    randomVariables['userLocation'] = True
    return randomVariables
    csvFile.close()


def collectWords(str):
    str = str.split()
    str2 = []
    # loop till string values present in list str
    for i in str:
        str2.append(i)

    return str2

def collectHashTags(finalList):
    hashTags = []
    for i in range(0, len(finalList)):
        if '#' in finalList[i]:
            hashTags.append(finalList[i])

    return hashTags

def printListOfTweets(tweetsList):
    for i in range(0, len(tweetsList)):
        print(
            "-----------------------------------------------------------------------------------------------------------")
        print("text mess---->  %s" % tweetsList[i].text)
        print("username---->  %s" % tweetsList[i].username)
        print("create---->  %s" % tweetsList[i].create)
        print("user location---->  %s" % tweetsList[i].location)
        print("user hashtags---->  %s" % tweetsList[i].hashtags)


def readRelatedWords(sentence):
    tempList = returnFirstColByRelatedWords()
    with open('datasets/relatedWords.csv', 'r', encoding='utf-8-sig') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            for i in range(0, len(tempList)):
                if row[i] in sentence and not row[i]=="":
                    return tempList[i]
                    csvFile.close()
    return ""
def readRelatedWordsDict(sentence,tempdictionary):
    tempList = returnFirstColByRelatedWords()
    with open('datasets/relatedWords.csv', 'r', encoding='utf-8-sig') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            for i in range(0, len(tempList)):
                if row[i] in sentence and row[i] not in "" and checkKey(tempdictionary,tempList[i]) == 1:
                  tempdictionary[tempList[i]] = True
                  break
    return tempdictionary
    csvFile.close()

def checkLocationInTweet(word):
    with open('datasets/relatedWords.csv', 'r', encoding='utf-8-sig') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            if row[3] in word or row[4] in word:
                return True
                csvFile.close()
    return False
    csvFile.close()


def printRandomVariables(randomVariables):
    for key, value in randomVariables.items():
        print(key,value)
    print("------------------")
    print("all random variables are %s" % randomVariables.keys().__len__())


def returnFirstColByRelatedWords():

    with open('datasets/relatedWords.csv', 'r', encoding='utf-8-sig') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            break
        return row
        csvFile.close()
def convertToSet(numpyArray):
    convertedArray=[]
    for i in range(0, len(numpyArray)):
        arr = []
        for [k, v] in numpyArray[i]:
            arr.append((k, v))
        convertedArray.append(arr)

    return convertedArray

def sortExamples(examples):
    for i in range(0, len(examples)):
        examples[i].sort()

    return examples