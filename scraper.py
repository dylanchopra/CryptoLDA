import math
import json
import requests
import itertools
import numpy as np
import time
import praw
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from datetime import datetime, timedelta
import multiprocessing

start_time=time.clock()

def make_request(uri, max_retries = 5):
    def fire_away(uri):
        response = requests.get(uri)
        assert response.status_code == 200
        return json.loads(response.content)
    current_tries = 1
    while current_tries < max_retries:
        try:
            time.sleep(.001)
            response = fire_away(uri)
            return response
        except:
            time.sleep(.001)
            current_tries += 1
    return fire_away(uri)


def pull_posts_for(subreddit, start_at, end_at):
    
    def map_posts(posts):
        return list(map(lambda post: {
            'id': post['id'],
            'created_utc': post['created_utc'],
            'prefix': 't4_'
        }, posts))
    
    SIZE = 1000
    URI_TEMPLATE = r'https://api.pushshift.io/reddit/search/submission?subreddit={}&after={}&before={}&size={}'
    
    post_collections = map_posts(make_request(URI_TEMPLATE.format(subreddit, start_at, end_at, SIZE))['data'])
    n = len(post_collections)
    while n == SIZE:
        last = post_collections[-1]
        new_start_at = last['created_utc'] - (10)
        more_posts = map_posts(make_request(URI_TEMPLATE.format(subreddit, new_start_at, end_at, SIZE))['data'])
        n = len(more_posts)
        post_collections.extend(more_posts)
    return post_collections

 
def  f(topics_dict,num,posts,subreddit,permno,number_of_processes):
    size=(int)(math.ceil((len(posts)/number_of_processes)))
    for i in range(size):
        if ((num*size)+i) >= len(posts):
            break
        submission_id=posts[(num*size)+i]
        submission=reddit.submission(id=submission_id)
        topics_dict["title"][i]=subreddit
        topics_dict["title"][i]=submission.title
        topics_dict["upvotes"][i]=submission.score
        topics_dict["number_of_comments"][i]=submission.num_comments
        timestamp=submission.created
        dt=datetime.fromtimestamp(timestamp)
        topics_dict["created"][i]=dt.date()
        topics_dict["body"][i]=submission.selftext
        submission.comments.replace_more(limit=None)
        current=""
        for comment in submission.comments.list():
            commentBody=(reddit.comment(comment)).body
            if(len(current+str(commentBody)+";") >= 32767):
                break
            current=current+str(commentBody)+";"
        topics_dict["comments"][i]=current
        print((num*size)+i)
    return topics_dict

config = {
        "username" : "Baguls",
        "client_id" : "4rjfeLCZn82bkA",
        "client_secret" : "vmKeEEGoYijQqyctqkOGAMkyEzI",
        "user_agent" : "Crypto Scraper"
    }
reddit = praw.Reddit(
        client_id = config['client_id'],
        client_secret = config['client_secret'],
        user_agent = config['user_agent']
    )


if __name__ == '__main__':

    df = pd.read_excel('input.xlsx')
    permnos=df['permno']
    subreddits=df['Subreddit']
    starts=df['Price Start']
    ends=df['Price End']
    topics_dictFinal = pd.DataFrame({ 
                    "permno":[],
                    "subreddit":[],
                    "title":[],
                    "upvotes":[],
                    "number_of_comments": [],
                    "created": [],
                    "body":[],
                    "comments":[]
                    })
    for j in range(len(subreddits)):

        subreddit=str(subreddits[j])
        if(subreddit=="nan"):
            continue
        subreddit=subreddit.split("/")
        subreddit=subreddit[1]
        print("Starting " + subreddit)
        permno=permnos[j]
        
        end_at=math.ceil(ends[j].timestamp())
        start_at=math.ceil(starts[j].timestamp())
        posts = pull_posts_for(subreddit, start_at, end_at)
        posts=np.unique([ post['id'] for post in posts ])
        posts=np.ndarray.tolist(posts)
        print(len(posts))
        number_of_processes=25
        if(len(posts) < 25):
            number_of_processes=1

        size=(int)(math.ceil((len(posts)/number_of_processes)))
        topics_dictA = { 
                    "permno":[permno]*size,
                    "subreddit":[subreddit]*size,
                    "title":[None]*size,
                    "upvotes":[None]*size,
                    "number_of_comments": [None]*size,
                    "created": [None]*size,
                    "body":[None]*size,
                    "comments":[None]*size
                    }
        pool=multiprocessing.Pool(number_of_processes)
        arguments=[]
        for x in range(number_of_processes):
            arguments.append((topics_dictA,x,posts,subreddit,permno,number_of_processes))
        results=pool.starmap(f,arguments)

        topics_dict=results[0]
        for i in range(number_of_processes-1):
            if(results[i+1]["title"]==None):
                break
            topics_dict["permno"]=topics_dict["permno"]+results[i+1]["permno"]
            topics_dict["subreddit"]=topics_dict["subreddit"]+results[i+1]["subreddit"]
            topics_dict["title"]=topics_dict["title"]+results[i+1]["title"]
            topics_dict["upvotes"]=topics_dict["upvotes"]+results[i+1]["upvotes"]
            topics_dict["number_of_comments"]=topics_dict["number_of_comments"]+results[i+1]["number_of_comments"]
            topics_dict["created"]=topics_dict["created"]+results[i+1]["created"]
            topics_dict["body"]=topics_dict["body"]+results[i+1]["body"]
            topics_dict["comments"]=topics_dict["comments"]+results[i+1]["comments"]
        topics_data = pd.DataFrame(topics_dict)
        frames=[topics_dictFinal,topics_data]
        topics_dictFinal=pd.concat(frames)
        del results
        del topics_data
        del topics_dict
        del posts
        del arguments
        print("Finished " + subreddit)
        print("Waiting 15 seconds before next subreddit")
        pool.close()
        pool.join()
        time.sleep(15)

    topics_dictFinal.to_csv('example.csv', index=False)
