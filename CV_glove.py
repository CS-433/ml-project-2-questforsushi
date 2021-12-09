from matplotlib.pyplot import plot
import numpy as np
from glove_template import glove
import pickle
def CV_glove(alpha_min, alpha_max,num,cooc_sets = None, COOC_SET_PATH = "Cooc_CV_seed1_k2.pkl",WORD_EMB_PATH = "testCV"): 
    if cooc_sets == None:
        with open(COOC_SET_PATH, 'rb') as f:
            cooc_sets = pickle.load(f)
    nmax = 100
    alphas = np.linspace(alpha_min,alpha_max, num = num)    
    x = []
    y = []
    avgLosses = []
    for alpha in alphas:
        avgLoss = 0
        for set in cooc_sets:
            WORD_EMB_PATH = "testCV_alpha%s"%(alpha)
            result = glove(20, alpha, set[0],SAVE_PATH=WORD_EMB_PATH)
            xs = result[0]
            ys = result[1]
            print(len(xs))
            testSet = set[1]
            for ix, jy, n in zip(testSet.row, testSet.col, testSet.data):
                # we use the logarithm of the co-occurrence values
                log_co = np.log(n/(len(cooc_sets)-1))
                fn = min(1, pow(n / nmax, alpha))
                x, y = xs[ix], ys[jy]
                avgLoss += fn * (log_co - np.dot(x, y))**2
            
        avgLosses.append(avgLoss/len(cooc_sets))
    plot(alphas,avgLosses)
    return alphas, avgLosses