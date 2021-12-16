import numpy as np
"""Returns BOW matrix with corresponding Y vector"""
def BOW_vector(vocab, POS_TWEET_PATH, NEG_TWEET_PATH):
    bows_list = []
    f = []
    for fn in [POS_TWEET_PATH,NEG_TWEET_PATH]:
            with open(fn) as file:
                with open(fn) as file:
                    f = file.readlines()
            bow = np.zeros((len(f),len(vocab))).astype('int8')
            for i in range(len(f)):
                    line = f[i]
                    # -1 is used instead of none for tokens not found in vocab
                    tokens = [vocab.get(t, -1) for t in line.strip().split()]
                    tokens = [t for t in tokens if t >= 0]
                    for token in tokens:
                            bow[i, token] += 1
            bows_list.append(bow)
    Xbow = np.concatenate((bows_list[0],bows_list[1]))
    Ypos = np.ones(bows_list[0].shape[0])
    Yneg = np.ones(bows_list[1].shape[0])*-1
    Y = np.concatenate((Yneg, Ypos))
    return Xbow, Y
"""Imports a generated embedding, 
file needs to be a txt file with the first element in each row being the element and
the rest being the word vector
vocab - dict of the words you want to import from the EMBEDDING
"""


def import_embedding(vocab, PATH_EMBEDDING):
    embedding = []
    with open(PATH_EMBEDDING, errors="ignore") as file:
                # line = file.readline()
                # lineArray = line.split()
                # 
                for line in file:
                        lineArray = line.split()
                        if len(embedding) == 0:
                                embedding = np.zeros((len(vocab),len(lineArray)-1))
                        wordI = vocab.get(lineArray[0], -1)
                      
                        if wordI > -1:
                                
                                embArray = np.array(lineArray[1:])
                               
                                embedding[wordI,:] = embArray
    np.save("imported_embeddings/glove.twitter.27B.200d.npy", embedding)
    return embedding
import csv
"""dont send in shuffled test data will fuck everything"""
def create_prediction(y_pred, name):
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for id,y in enumerate(y_pred):
            writer.writerow({'Id':int(id+1),'Prediction':int(y)})
def create_noID_data():
    with open("twitter-datasets/test_data.txt", 'rb') as f:
        text = f.readlines()
        cleaned_text = []
        for i,line in enumerate(text):
            line = str(line)
            temp = line.split(",",1)
            #temp[1] += '\n'
            cleaned_text.append( bytes(temp[1][0:-3] + '\n',"utf-8"))
    print(text[0])       
    with open("twitter-datasets/test_data_noId.txt", 'wb') as f:
        f.writelines(cleaned_text)