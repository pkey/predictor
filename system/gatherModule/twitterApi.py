# Prepare Twitter API TODO hide the keys or regenerate them
from twython import Twython
import keys as keys

twitterAccess = Twython(keys.APP_KEY, keys.APP_SECRET, oauth_version=2)
ACCESS_TOKEN = twitterAccess.obtain_access_token()

twitter = Twython(keys.APP_KEY, access_token=ACCESS_TOKEN)


# Actual code
def search(query):
    return twitter.search(q=query, language="EN")['statuses'];
