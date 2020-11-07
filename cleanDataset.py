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
from functions import *
Headers = ["Tweets", "Username", "Created_at", "User location", "Coordinates"]
randomVariables = {}
analyzer = SentimentIntensityAnalyzer()
# diavazw to text arxeio opou einai oi trained rules kai ta facts
with open('models/model.txt') as f:
   content = f.readlines()
   content = [x.strip() for x in content]

tweetsList = []
tweetsList=readCSV('train3last900.csv')
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

csvFile1 = open('datasets/clear.csv', 'w', encoding='utf-8-sig', newline='')
csvWriter = csv.writer(csvFile1)
csvWriter.writerow(Headers)
clearTweets=[]
# Headers = ["Tweets", "Username", "Created_at", "User location", "Coordinates"]
for i in range(1,len(tweetsList)):
    evidenceDict={}
    evidenceDict = randomVariables.copy()
    readRelatedWordsDict(tweetsList[i].text,evidenceDict)
    tweet1 = NewTweet(tweetsList[i].text, tweetsList[i].create, tweetsList[i].location)
    if evidenceDict['location_in_tweet']==True:
        clearTweets.append(Tweet(tweetsList[i].text,tweetsList[i].username,tweetsList[i].create, tweetsList[i].location,
                      "-"))

for i in range(0,len(clearTweets)):
    csvWriter.writerow([clearTweets[i].text,clearTweets[i].username,clearTweets[i].create,clearTweets[i].location,
                        "-"])
csvFile1.close()