import numpy as np
import pickle as pickle
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
"""Creates file with test data without ids for input in other functions"""
def create_noID_data():
    with open("twitter-datasets/test_data.txt", 'rb') as f:
        text = f.readlines()
        cleaned_text = []
        for i,line in enumerate(text):
            line = str(line)
            temp = line.split(",",1)
            #temp[1] += '\n'
            cleaned_text.append( bytes(temp[1][0:-3] + '\n',"utf-8"))    
    with open("twitter-datasets/test_data_noId.txt", 'wb') as f:
        f.writelines(cleaned_text)
"""
Takes a word embedding array and returns an embedding of each tweet, as the AVERAGE vector of every word it contains
:param WORD_EMBEDDING_PATH: path of array containing the word embedding
:param tweet_PATH: path of the tweets to be embedded
:return: embedding of the tweets, as average embedding of the words
"""
def build_tweet_vector(WORD_EMBEDDING_PATH, VOCAB_PATH, tweet_PATH, SAVE_PATH,   TFID_weighting = False):
   
    
    # contains the embedding for each word, as a dim dimensional vector
    word_embedding = np.load(WORD_EMBEDDING_PATH)
    DIM = word_embedding.shape[1]
    print(DIM)

    # vocab is a dict from textual word to its numerical position
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    with open(tweet_PATH, 'rb') as pos_file:
        tweets = pos_file.read().splitlines()
    with open('TFID_stopW.pkl', 'rb') as f:
        TFID = pickle.load(f)

    # ex : a = str(pos_set[0].split()[1])[2:-1]
    # number of positive tweets
    counter = 0
    embedded_tweets = np.empty([len(tweets), DIM])
    nb_tweets = len(tweets)
    if TFID_weighting:
        TF_dict = dict(zip(TFID.get_feature_names_out(), TFID.idf_))
    for i in range(nb_tweets):
        """prints progress"""
        if i%100000 == 0: print(i)
        tweet = str(tweets[i])
        tweet_words = tweet.split()
        tweet_encoding = np.zeros(DIM)
        number_of_words = 0
       
        
        totWeights = 0
        for l in range(len(tweet_words)):
            word = tweet_words[l]
            word_index = vocab.get(str(word))
            if TFID_weighting:
                """sees if the data was in the TFID vocab"""
                weight = TF_dict.get(str(word))
           
            """checks so that it is in our vocab """
            if TFID_weighting and  word_index is not None and weight is not None:
                tweet_encoding += word_embedding[word_index]*weight
                totWeights += weight
                number_of_words += 1
        
            if not TFID_weighting and word_index is not None: 
                tweet_encoding += word_embedding[word_index]
                number_of_words += 1
                
                    
                
        if number_of_words != 0 and not TFID_weighting :
            tweet_encoding = tweet_encoding / number_of_words
        elif number_of_words != 0:
            tweet_encoding = tweet_encoding / number_of_words/totWeights
            
                
        embedded_tweets[i, :] = tweet_encoding
    print(counter)
    np.save(SAVE_PATH, embedded_tweets)
    return embedded_tweets
"""Trains glove embedding, can use intitial condition provding paht to embedding and with_initial = True"""
def glove(embedding_dim=25, alpha=3 / 4,cooc = None, COOC_PATH = 'cooc.pkl',SAVE_PATH = "Embeddings/embeddings",with_intial = False, INITIAL_PATH = "imported_embeddings/glove.twitter.27B.25d.npy"):
    print("loading cooccurrence matrix")
    if cooc == None:
        with open(COOC_PATH, 'rb') as f:
            cooc = pickle.load(f)
        print("{} nonzero entries".format(cooc.nnz))
    
    nmax = 100  # this will serve as a cut-off to do not exagerate the importance of some very popular words
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())
     
    print("initializing embeddings")
    xs = []
    ys = []
    if with_intial:
        xs = np.load(INITIAL_PATH)
        ys = np.copy(xs)
    else:
        xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
        ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001  # gradient descent

    epochs =5

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            # we use the logarithm of the co-occurrence values
            log_co = np.log(n)
            fn = min(1, pow(n / nmax, alpha))
            x, y = xs[ix], ys[jy]
            grad_common_part = 2 * fn * (log_co - np.dot(x, y))
            xs[ix] += eta * grad_common_part * y
            ys[jy] += eta * grad_common_part * x

    np.save(SAVE_PATH + str(nmax) + "_dim" + str(embedding_dim), xs)
    return (xs, ys)
