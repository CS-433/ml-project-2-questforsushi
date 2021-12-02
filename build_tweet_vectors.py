import numpy as np
import pickle

# Opens stored data-----------------------------------------------------------------------------------------------------
def Build_tweet_vector(WORD_EMBEDDING_PATH, tweet_PATH, SAVE_PATH):
    

    # contains the embedding for each word, as a dim dimensional vector
    word_embedding = np.load(WORD_EMBEDDING_PATH)
    DIM = word_embedding.shape[1]
    print(DIM)

    # vocab is a dict from textual word to its numerical position
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    with open(tweet_PATH, 'rb') as pos_file:
        tweets = pos_file.read().splitlines()
        
 

    # ex : a = str(pos_set[0].split()[1])[2:-1]
    # number of positive tweets

    embeded_tweets = np.empty([len(tweets),DIM])
    nb_tweets = len(tweets)
    for i in range(nb_tweets):
        tweet = tweets[i]
        tweet_words = tweet.split()
        tweet_encoding = np.zeros(DIM)
        number_of_words = 0
        for word in tweet_words:
            # we remove the part of the "words" that is not part of them and get the corresponding index
            word_index = vocab.get(str(word)[2:-1])
            if word_index is not None:
                tweet_encoding += word_embedding[word_index]
                number_of_words += 1
        if number_of_words != 0:
            tweet_encoding = tweet_encoding/number_of_words
        #TODO : FIND A WAY TO ADD THE NEW VECTOR TO EMBEDED TWEET EFFICIENTLY
        embeded_tweets[i,:] =  tweet_encoding
  
        
    
    np.save(SAVE_PATH+str(DIM), embeded_tweets)
    return embeded_tweets
    #np.save(NEG_SAVE_PATH+str(DIM), embedded_neg)

def main():
    WORD_EMBEDDING_PATH = "Embeddings/embeddings_nmax100_dim20.npy"
    POS_SET_PATH = "twitter-datasets/train_pos_full.txt"
    POS_SAVE_PATH = "Embedded_tweets/embeddings_pos_dim"
    NEG_SET_PATH = "twitter-datasets/train_neg_full.txt"
    NEG_SAVE_PATH = "Embedded_tweets/embeddings_neg_dim"
    Build_tweet_vector(WORD_EMBEDDING_PATH, POS_SET_PATH, POS_SAVE_PATH)
    Build_tweet_vector(WORD_EMBEDDING_PATH, NEG_SET_PATH, NEG_SAVE_PATH)
    #TODO : Append each tweet encoding to a big array
if __name__ == '__main__':
    main()