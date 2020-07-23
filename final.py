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
import re
import string
import nltk
#nltk.download('punkt')
from gensim import corpora, models, similarities 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
import pyLDAvis.gensim
import os.path
from os import path
from multiprocessing import Process, Queue, Manager

def initial_clean(text):
    """
    Function to clean text-remove punctuations, lowercase text etc.    
    """
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower() # lower case text
    text = nltk.word_tokenize(text)
    return text

stop_words = stopwords.words('english')
stop_words.extend(['reddit', 'nan', 'nexus', 'nxs', 'https', 'twitter' 'bitcoin', 'crypto', 'say','use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do','took','time','year', 'for', 'so', 'to', 'and',
'done', 'try', 'many', 'some','nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line','even', 'also', 'may', 'take', 'come', 'new','said', 'like','people'])

def remove_stop_words(text):
    words=[word for word in text if word not in stop_words]
    remove_words=["Twitter","delete","remove","twitter","status/","https","tweet", "nan"]
    words=[word for word in words if not any(remove_words in word for remove_words in remove_words)]
    return words

def stem_words(text):
    """
    Function to stem words
    """
    stemmer = PorterStemmer()
    try:
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if len(word) > 1] # no single letter words
    except IndexError:
        pass
    return text 

def apply_all(text):
    """
    This function applies all the functions above into one
    """
    return stem_words(remove_stop_words(initial_clean(text)))

if __name__ == '__main__':
    df = pd.read_excel('input.xlsx')
    permnos = df['permno']
    subreddits = df['Subreddit']
    starts = df['Price Start']
    ends = df['Price End']
    dicts = []

    startInt=1004
    numSubreddits=924


    for r in range(numSubreddits):
        currentInt=startInt+r
        if(currentInt >= len(subreddits)):
            break
        subreddit=str(subreddits[currentInt])
        permno=permnos[currentInt]
        if(subreddit=="nan"):
            continue
        subreddit=subreddit.split("/")
        subreddit=subreddit[1]
        df=pd.read_csv('all(secondhalf).csv',error_bad_lines=False)
        df=df.loc[df['subreddit'] == subreddit]
        #df=pd.read_csv('steemdollars.csv',error_bad_lines=False)
        numDocuments=len(df['comments'])
        print("subreddit: "+str(subreddit)+"  number of documents: "+str(numDocuments))
        if(numDocuments<50):
            continue
        df['comments'] = df['comments'].astype(str) 
        df['title'] = df['title'].astype(str) 
        df['comments']=df['title'].str.cat(df['comments'],sep=" ")
        df['tokenized_comments'] = df['comments'].apply(apply_all)    
        #Create a Gensim dictionary from the tokenized data 
        tokenized = df['tokenized_comments']
        #Creating term dictionary of corpus, where each unique term is assigned an index.
        dictionary = corpora.Dictionary(tokenized)
        #Filter terms which occurs in less than 1 review and more than 80% of the reviews.
        dictionary.filter_extremes(no_below=1, no_above=0.80)
        #convert the dictionary to a bag of words corpus 
        corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
        #print(corpus[:1])

        termFreq=[[(dictionary[id], freq) for id, freq in cp] for cp in corpus]
        pumpCount=0
        dumpCount=0
        scamCount=0
        for f in range(len(termFreq)):
            for c in range(len(corpus[f])):
                if("pump" in termFreq[f][c]):
                    pumpCount+=1
                if("dump" in termFreq[f][c]):
                    dumpCount+=1
                if("scam" in termFreq[f][c]):
                    scamCount+=1

        #LDA
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 20, id2word=dictionary, passes=25)
        ldamodel.save('model_combined.gensim')
        topics = ldamodel.print_topics(num_words=15)
        topicStore=""
        for topic in topics:
            topicStore=topicStore+str(topic)+"\n"

        #get_document_topics = ldamodel.get_document_topics(corpus[0])
        #print(get_document_topics)

        numPump=0
        numDump=0
        numBoth=0
        pumpTopics=[]
        dumpTopics=[]
        bothTopics=[]
        weights=[0.0]*20
        for i,topic in ldamodel.show_topics(formatted=True, num_topics=20, num_words=15):
            if("pump" in topic):
                numPump+=1
                pumpTopics.append(i)
            if("dump" in topic):
                numDump+=1
                dumpTopics.append(i)
            if("dump" in topic and "pump" in topic):
                numBoth+=1
                bothTopics.append(i)
            print(str(i)+": "+ topic)
            print()
            topic=str(topic)
            topics=topic.split(" + ")
            totalweight=0.0
            for t in topics:
                s=t.split("*")
                w=float(s[0])
                totalweight+=w
            for t in topics:
                s=t.split("*")
                w=float(s[0])
                word=str(s[1].split('"')[1])
                if(word in ["pump","dump"]):
                    weights[i]+=(w/totalweight)
        avgrelativeweight=0.0
        wcount=0
        for weight in weights:
            if(weight>0):
                avgrelativeweight+=weight
                wcount+=1
        if(wcount>0):
            avgrelativeweight=avgrelativeweight/wcount

        print("numPump:"+str(numPump)+"\n")
        print("numDump:"+str(numDump)+"\n")
        print("numBoth:"+str(numBoth)+"\n")
        print("pumpTopics:"+str(pumpTopics)+"\n")
        print("dumpTopics:"+str(dumpTopics)+"\n")
        print("bothTopics:"+str(bothTopics)+"\n")
        totaloccurrence=0
        totaloccurrence10=0
        totaloccurrence25=0
        totaloccurrence50=0
        w=0.0
        w10=0.0
        w25=0.0
        w50=0.0
        # [document,topic,percentage]
        percentages=[]
        for x in range(numDocuments):
            distrib=ldamodel[corpus[x]]
            matchingTopics=[]
            for y in range(len(distrib)):
                if((distrib[y][0] in bothTopics) or (distrib[y][0] in pumpTopics) or (distrib[y][0] in dumpTopics)):
                    matchingTopics.append((distrib[y][1],distrib[y][0]))
            # Need to only use the maximum probability topic to avoid double counting
            if(len(matchingTopics)>0):
                totaloccurrence+=1
                maxTopic=max(matchingTopics)
                percentages.append([x,maxTopic[1],maxTopic[0]])
                if(maxTopic[0]>0.0945):
                    totaloccurrence10+=1
                if(maxTopic[0]>0.2445):
                    totaloccurrence25+=1
                if(maxTopic[0]>0.4945):
                    totaloccurrence50+=1
        pTotal=str(round(totaloccurrence/numDocuments,5))
        p10=str(round(totaloccurrence10/numDocuments,5))
        p25=str(round(totaloccurrence25/numDocuments,5))
        p50=str(round(totaloccurrence50/numDocuments,5))
        print("Percentage of total posts displaying a topic involving pump & dump: "+ pTotal)
        print("Percentage of total posts with a greater than 10 percent chance of identifying to a topic involving pump & dump: "+ p10)
        print("Percentage of total posts with a greater than 25 percent chance of identifying to a topic involving pump & dump: "+ p25)
        print("Percentage of total posts with a greater than 50 percent chance of identifying to a topic involving pump & dump: "+ p50)

        """ for z in range(len(percentages)):
            print(percentages[z]) """

        scamRelatedTerms=pumpCount+dumpCount+scamCount

        output = pd.DataFrame({
            "permno": [None]*1,
            "subreddit": [None]*1,
            "numDocuments": [None]*1,
            "pTotal": [None]*1,
            "p10": [None]*1,
            "p25": [None]*1,
            "p50": [None]*1,
            "avgrelativeweight": [None]*1,
            "scamRelatedTerms": [None]*1,
            "topics": [None]*1
            })

        output["permno"][0]=permno
        output["subreddit"][0]=subreddit
        output["numDocuments"][0]=numDocuments
        output["pTotal"][0]=pTotal
        output["p10"][0]=p10
        output["p25"][0]=p25
        output["p50"][0]=p50
        output["avgrelativeweight"][0]=avgrelativeweight
        output["scamRelatedTerms"][0]=scamRelatedTerms
        output["topics"][0]=topicStore

        outputFileExists=path.exists("results.csv")
        if(outputFileExists):
            of=pd.read_csv('results.csv',error_bad_lines=False)
            frames=[of,output]
            finalData=pd.concat(frames)
            finalData.to_csv('results.csv', index=False)

        else:
            output.to_csv('results.csv', index=False)

