import tweepy  # for tweets etc.
import csv  # Import csv
import time
import os  # for delete file
import datetime

# keys για να εχουμε προσβαση στο Twitter, ωστε να κανουμε κλησεις προς αυτο
consumer_key = '5IvJmMUSAF9Cer5X8DKpzgHfB'
consumer_secret = 'Ag7CJzrFuiv28cPjyoh0Zg4QqUGdI27aGzQSPWNC7ESmiSxBWP'
access_token = '1160447126402453506-v41YtKRaV95rWw7KmixcvjzmFXsBfq'
access_token_secret = 'mpj4eJKQmCu55fqqjTadlA3XDtEVJUkqiaE0zwnIqpUnt'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
existCoordinates = False
#εδω δημιουργειται η πρωτη γραμμη, με το τιτλο της καθε στηλης
Headers = ["Tweets", "Username", "Created_at", "User location", "Coordinates"]
#κανω κληση προς το twitter με τα παρακατω hashtags
hashtags = ["#vacation"]

#εδω δημιουργω το αρχειο που θα αποθηκευτουν τα tweets
if os.path.exists('negative1.csv'):
    os.remove('negative1.csv')

csvFile = open('negative1.csv', 'w', encoding='utf-8-sig', newline='')
csvWriter = csv.writer(csvFile)
api = tweepy.API(auth)

counter = 0
csvWriter.writerow(Headers)
#με την παρακατω επαναληψη κανω κληση προς το API του Twitter για να βρω ολα τα tweets απο το 2019-01-05
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