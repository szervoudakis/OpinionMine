import tweepy  # for tweets etc.
import csv  # Import csv
import time
import os  # for delete file
import datetime


consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
existCoordinates = False

Headers = ["Tweets", "Username", "Created_at", "User location", "Coordinates"]
#put in list the hashtag #vacation
hashtags = ["#vacation"]


if os.path.exists('negative1.csv'):
    os.remove('negative1.csv')

csvFile = open('negative1.csv', 'w', encoding='utf-8-sig', newline='')
csvWriter = csv.writer(csvFile)
api = tweepy.API(auth)

counter = 0
csvWriter.writerow(Headers)

print('-------------------------------------tweets by hastag---------------------------------------------------------')
for i in range(len(hashtags)):
    tweets = tweepy.Cursor(api.search, q=hashtags[i],lang="en").items(300)
    for tweet in tweets:
        print('-----------------------------------------')
        existCoordinates = False
        if (tweet.place is not None):
            existCoordinates = True
        if (existCoordinates == True):
            csvWriter.writerow([tweet.text, tweet.user.screen_name, tweet.created_at, tweet.user.location,tweet.place.bounding_box.coordinates])
        if (existCoordinates == False):
            csvWriter.writerow([tweet.text, tweet.user.screen_name, tweet.created_at, tweet.user.location,'-'])
csvFile.close()