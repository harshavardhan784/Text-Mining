import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import bs4
import textblob
import openpyxl
from pathlib import Path
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# Funtion for polarity sentiment
def analyse_sentiment(tweet):
    analysis=TextBlob(tweet)
    if analysis.sentiment.polarity>0:
        return 'Positive Score'
    elif analysis.sentiment.polarity==0:
        return 'Neutral Score'
    else:
        return 'Negative Score'

# Funtion for subjectiving sentiment
def subject_sentiment(tweet):
    analysis=TextBlob(tweet)
    return analysis.subjectivity

# Function for syllable count
def countSyllables(word):
    vowels = "aeiouy"
    numVowels = 0
    lastWasVowel = False
    for wc in word:
        foundVowel = False
        for v in vowels:
            if v == wc:
                if not lastWasVowel: numVowels+=1   #don't count diphthongs
                foundVowel = lastWasVowel = True
                break
        if not foundVowel:  #If full cycle and no vowel found, set lastWasVowel to false
            lastWasVowel = False
    if len(word) > 2 and word[-2:] == "es": #Remove es - it's "usually" silent (?)
        numVowels-=1
    elif len(word) > 1 and word[-1:] == "e":    #remove silent e
        numVowels-=1
    return numVowels

# Reading links from given input file

dataframe2 = pd.read_excel('Input.xlsx')
dataframe1=dataframe2

# dataframe1 = {'URL': [ 'https://insights.blackcoffer.com/ai-in-healthcare-to-improve-patient-outcomes/','https://insights.blackcoffer.com/what-if-the-creation-is-taking-over-the-creator/']}
# dataframe1 = pd.DataFrame(dataframe1)

# Initializing everything to zero

dataframe1['POSITIVE Score']=0
dataframe1['NEGATIVE Score']=0
dataframe1['POLARITY Score']=0
dataframe1['SUBJECTIVITY SCORE']=0
dataframe1['AVG SENTENCE LENGTH']=0
dataframe1['PERCENTAGE OF COMPLEX WORDS']=0
dataframe1['FOG INDEX']=0
dataframe1['AVG NUMBER OF WORDS PER SENTENCE']=0
dataframe1['COMPLEX WORD COUNT']=0
dataframe1['WORD COUNT']=0 
dataframe1['SYLLABLE PER WORD']=0
dataframe1['PERSONAL PRONOUNS']=0 
dataframe1['AVG WORD LENGTH']=0 

# Initialize i to 0
i=0
for x in dataframe1['URL']:
    text=requests.get(x)
    Text=BeautifulSoup(text.content)
    Text=Text.getText(strip=True)
    Text=re.sub(r'\[d+\]',"",Text)
    Text=re.sub('[0-9]+',"",Text)
    sentence=sent_tokenize(Text)
    sentence=pd.DataFrame(sentence)
    sentence.columns=['sentence']
    sentence['sentiment']=[str(analyse_sentiment(x)) for x in sentence.sentence]
    sentence['subjectivity']=[str(subject_sentiment(x)) for x in sentence.sentence]
    
    
    sum = 0
    for x in sentence.sentence:
        analysis=TextBlob(x)
        ele=float(analysis.sentiment.polarity)
        sum += ele
    res = sum / len(sentence.sentence)
    
    POLARITY_SCORE=res
    dataframe1.at[i,'POLARITY Score']=POLARITY_SCORE
    
    # counting positive and negative scores
    Positive_Score=0
    Neutral_Score=0
    Negative_Score=0
    
    for x in sentence.sentiment:
        if x=='Positive Score':
            Positive_Score +=1
        if x=='Neutral Score':
            Neutral_Score +=1
        if x=='Negative Score':
            Negative_Score +=1    
            
    sent=len(sentence)
    # Positive_Score,Neutral_Score,Negative_Score=sentence.sentiment.value_counts()
    dataframe1.at[i,'POSITIVE Score']=Positive_Score
    # dataframe1.at[i,'Neutral Score']=Neutral_Score
    dataframe1.at[i,'NEGATIVE Score']=Negative_Score
    
    # avg value of subjectivity 
    sum=0
    for x in sentence.sentence:
        ele=float(subject_sentiment(x))
        sum += ele
    subjectivity = sum / len(sentence.sentence)
    dataframe1.at[i,'SUBJECTIVITY SCORE']=subjectivity
    textwords=word_tokenize(Text)

    dataframe1.at[i,'WORD COUNT']=len(textwords)
    
    # To find no of complex words 
    count = 0
    for myword in textwords:
        d = {}.fromkeys('aeiou',0)
        haslotsvowels = False
        for x in myword.lower():
            if x in d:
                d[x] += 1
        for q in d.values():
            if q > 2:
                haslotsvowels = True
        if haslotsvowels:
            count += 1      
    ComplexWords=count
    dataframe1.at[i,'COMPLEX WORD COUNT']=ComplexWords
    
    # PERCENTAGE OF COMPLEX WORDS
    percent=count/len(textwords)
    Percentage=percent*100
    dataframe1.at[i,'PERCENTAGE OF COMPLEX WORDS']=Percentage
    
    # PERSONAL PRONOUNS
    pronounRegex = re.compile(r'\bI\b|\bwe\b|\bWe\b|\bmy\b|\bMy\b|\bours\b|\bus\b')
    list=[]
    list=pronounRegex.findall(Text)
    PersonalPronouns=len(list)
    dataframe1.at[i,'PERSONAL PRONOUNS']=PersonalPronouns
    
    # Syllable Per Word
    for x in textwords:
        sum = 0
        ele=float(countSyllables(x))
        sum += ele
    res = sum / len(textwords)
    SyllablePWord=res
    dataframe1.at[i,'SYLLABLE PER WORD']=SyllablePWord
    
    # AVG SENTENCE LENGTH
    AvgWord=len(textwords)/len(sentence)
    dataframe1.at[i,'AVG SENTENCE LENGTH']=AvgWord
    textwords=[word for word in textwords if word.isalnum()]
    stop_words=set(stopwords.words('english'))
    textwords=[word for word in textwords if not str.lower(word) in stop_words]
    wordFreq=FreqDist(textwords)
    
    # FOG Index
    fog=(0.4*(len(textwords)/sent)+100*(ComplexWords/len(textwords)))
    dataframe1.at[i,'FOG INDEX']=fog
    
    # AVG NUMBER OF WORDS PER SENTENCE
    dataframe1.at[i,'AVG NUMBER OF WORDS PER SENTENCE']=len(textwords)/len(sentence.sentence)
    
    # Avg word length in integer
    sum=0
    for word in textwords:
        sum=sum+len(word)
    average =sum/len(textwords)
    dataframe1.at[i,'AVG WORD LENGTH']=(average)
    
    # increment i for next url
    i=i+1   
    
# Data Frame dataframe1 contains all the variables
file_name = 'Output Data Structure.xlsx'
  
# saving the excel
dataframe1.to_excel(file_name)
print(dataframe1)