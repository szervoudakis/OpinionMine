class NewTweet:
    def __init__(self, text, created, location):
        self.text = text
        self.created = created
        self.location = location

#class for tweets with properties text,username, creation date, location and hashtags
class Tweet:
    def __init__(self, text, username, create, location, hashtags):
        self.text = text
        self.username = username
        self.create = create
        self.location = location
        self.hashtags = hashtags
