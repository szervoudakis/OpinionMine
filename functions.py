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
#this file contains all functions that used of our system

#read csv file row by row and store content in list
def readCSV(nameOfFile):
    tweetsList = []
    nameOfFile="datasets/"+nameOfFile
    #
    with open(nameOfFile, 'r', encoding='utf-8-sig') as csvFile:
        reader = csv.reader(csvFile)

        count=0
        for row in reader:
            count=count+1
            temp = collectWords(row[0])
            hashtags = collectHashTags(temp)
            tweetsList.append(
                Tweet(row[0], row[1], row[2], row[3],
                      hashtags))
    csvFile.close()

    return tweetsList
#function that takes as input text, analyser (vader tool)  and the current randomvariables
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

#check if key exists in dict
def checkKey(dict, key):
    if key in dict.keys():
        return 1
    else:
        return 0
#return the score of text (sentiment analysis)
def sentiment_analyzer(tweetText, analyser):
    score = analyser.polarity_scores(tweetText)
    lb = score['compound']
    if lb >= 0.05:
        return 1
    elif (lb > -0.05) and (lb < 0.05):
        return 0
    else:
        return -1
#check if text contains some location of Crete
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
#collect all hashtag from list
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

#if some related word - synonym word exist in a text , return the category (random variable)
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
#if in text exist some synonym word return put the True value in the random variable
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
#if in text exist some location of Crete, return true
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

#return all categories from csv file
def returnFirstColByRelatedWords():

    with open('datasets/relatedWords.csv', 'r', encoding='utf-8-sig') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            break
        return row
        csvFile.close()
#convert array to list
def convertToSet(numpyArray):
    convertedArray=[]
    for i in range(0, len(numpyArray)):
        arr = []
        for [k, v] in numpyArray[i]:
            arr.append((k, v))
        convertedArray.append(arr)
    return convertedArray
#sort the examples
def sortExamples(examples):
    for i in range(0, len(examples)):
        examples[i].sort()

    return examples