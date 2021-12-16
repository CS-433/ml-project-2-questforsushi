import numpy as np
import pickle

from numpy.lib.npyio import save


# Opens stored data-----------------------------------------------------------------------------------------------------
def build_tweet_vector(WORD_EMBEDDING_PATH, VOCAB_PATH, tweet_PATH, SAVE_PATH,  pos_or_neg = "", TFID_weighting = False):
    """
    Takes a word embedding array and returns an embedding of each tweet, as the AVERAGE vector of every word it contains
    :param WORD_EMBEDDING_PATH: path of array containing the word embedding
    :param tweet_PATH: path of the tweets to be embedded
    :param SAVE_PATH: TODO : Change this, we want a single path for both
    :return: embedding of the tweets, as average embedding of the words
    """
    
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
        if i%100000 == 0: print(i)
        tweet = str(tweets[i])
        tweet_words = tweet.split()
        tweet_encoding = np.zeros(DIM)
        number_of_words = 0
       
        
        totWeights = 0
        for l in range(len(tweet_words)):
            word = tweet_words[l]
            """sees if the data was in the TFID vocab"""
            # if TFID.transform([word]).data.size == 0:
            #     continue
            # we remove the part of the "words" that is not part of them and get the corresponding index
            word_index = vocab.get(str(word))
            if TFID_weighting:
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


def main():
    WORD_EMBEDDING_PATH = "Embeddings/embeddings_nmax100_dim20.npy"
    VOCAB_PATH = "vocab_full.pkl"
    POS_SET_PATH = "twitter-datasets/train_pos_full.txt"
    NEG_SET_PATH = "twitter-datasets/train_neg_full.txt"
    SAVE_PATH = "Embedded_tweets"

    build_tweet_vector(WORD_EMBEDDING_PATH, VOCAB_PATH, POS_SET_PATH, SAVE_PATH, "pos")
    build_tweet_vector(WORD_EMBEDDING_PATH, VOCAB_PATH, NEG_SET_PATH, SAVE_PATH, "neg")
    # TODO : Append each tweet encoding to a big array and output a single one. See if we want the 1/0 col to appear inside too


if __name__ == '__main__':
    main()
