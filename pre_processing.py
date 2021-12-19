import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
import string
import nltk
nltk.download('wordnet') 
from textblob import TextBlob
import nltk
from nltk.stem import WordNetLemmatizer
from scipy.sparse import *



# Remove special characters except for ! and ?
Puntuations = "@#$%^&*()_+-=,./<>;':\|"
with open("train_neg.txt", "r") as f:
    with open("new.txt", "w") as n:
        no_punts = []
        for line in f:
            for char in line:       
                if(char not in Puntuations):
                    #no_punts[line,:] =  char
                    n.write(char)
n.close()
f.close()




# Remove duplicate lines
with open('new.txt') as result:
        uniqlines = set(result.readlines())
        with open('new2.txt', 'w') as rmdup:
            rmdup.writelines(set(uniqlines))
   
        
   
    
   
    
   
# Lemmatization
lemmatizer = WordNetLemmatizer()
with open("new3.txt", "w") as n:
    with open('new2.txt', "r") as k: 
        for line in k:
            n.write(str("\n"))
            d = line.split()
            for word in d:
                x = lemmatizer.lemmatize(word)
                n.write(x+str(" "))





# Correct misspelled words, needs to install textblob
# it is not very good because it confuses some word with others (e.g. caaats = cards)
with open("new_fianal.txt", "w") as n:
    with open("new3.txt", "r") as f:
        for line in f:   
            x = line
            textBlb = TextBlob(x)            # Making our first textblob
            textCorrected = textBlb.correct() # Correcting the text
            x = str(textCorrected)
            n.write(x)




            
            
                        

