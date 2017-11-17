#!/usr/bin/env python
# ==============================================================================
# This is a HMM decoding problem.
# Given the probability of transmission and emission, the decoding task is to
# find the hidden states that maximize the probability.
# Specifically, in our case where we are trying to tag the speaker
# of each sentence in a dialogue.
# ==============================================================================

from __builtin__ import str
import random
import math
from collections import defaultdict
from random import randint
import sys
import operator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import numpy as np
import io


def preprocess(movie_idx):
    '''
    preprocess movie lines
    '''
    cid = {}
    lines = []
    batch_size = 10
    n_char = 0

    file = io.open('./../cornell movie-dialogs corpus/movie_lines.txt',
                   encoding = "ISO-8859-1")
   # file = open('./cornell movie-dialogs corpus/movie_lines.txt', 'r')
    for line in file:
        f = line.rstrip().split(' +++$+++ ')
        movieid = int(f[2].lstrip('m'))
        if movieid < movie_idx:
            continue
        if movieid > movie_idx:
            break

        if len(f) >= 5: # filter out line without words
            # count user, assign interger user id from 0
            if f[1] not in cid:
                cid[f[1]] = n_char
                n_char = n_char + 1

            i = int(f[0][1:]) # line id
            lines.append([i, f[1], f[4]]) # keep line id, user id and text
    file.close()
    # replace user id
    for i in lines:
        i[1] = cid[i[1]]
    # sorts in place based on line id
    lines.sort(key=lambda tup: tup[0], reverse=False)
    # process words
    tokenizer = RegexpTokenizer(r'\w+')
    stop = stopwords.words('english')
    wordnet_lemmatizer = WordNetLemmatizer()
    alltokens = [] # a mixture of all tokens
    for i in lines:
        tokens = tokenizer.tokenize(i[2].lower())
        newtokens = []
        for t in tokens:
            newt = wordnet_lemmatizer.lemmatize(t, pos='v')
            newtokens.append(newt)
        tokens = newtokens
        newtokens = []
        for t in tokens:
            newt = wordnet_lemmatizer.lemmatize(t, pos='n')
            newtokens.append(newt)
        tokens = newtokens
        tokens = [t for t in tokens if t not in stop]
        i[2] = tokens
        alltokens += tokens

    # generate vocabulary
    n_alltokens = len(set(alltokens))
    print "total tokens : %d"%n_alltokens
    n_vocab = int(n_alltokens * 0.7)
    print "choose vocabulary size : %d"%n_vocab
    result = nltk.FreqDist(alltokens)
    token_freq = result.most_common(n_vocab)
    # print token_freq
    vocab = [w for (w, _) in token_freq]
    # vocab_cnt = [c for (_, c) in token_freq]

    # generate BOW for each line
    bow = np.zeros((len(lines), n_vocab + 1 ))
    bow = bow.astype(int)
    try:
        for i in range(len(lines)):
            for j in lines[i][2]:
                try:
                    word_idx = vocab.index(j)
                except ValueError:
                    word_idx = n_vocab # idx of OTHER words not in the most freq vocab
                bow[i][word_idx] += 1
    except IndexError:
        print "IndexError"
        sys.exit()
    return lines, vocab, bow, n_char


def main(movie_idx):
    '''
    preprocess movie lines
    assume a fixed dialog length
    try different diaglog segmentation
    '''
    lines, vocab, bow, n_char = preprocess(movie_idx)

    batch_size = 10 # assumed length of each dialog
    nb_accu = []
    hmm_accu = []
    batch_offset = 0

#     try different dialog segmentation
    batch_start_idx = range(batch_offset,
                            len(lines) - 2 * batch_size,
                            batch_size)
    num_batch = int(len(batch_start_idx) / 5)
    random.seed(0)
    random.shuffle(batch_start_idx)
    for batch_offset in range(10):
        batch_start_idx_shifted = [i + batch_offset for i in batch_start_idx]
        nb_accu_avg, hmm_accu_avg = evaluate(lines,
                                             vocab,
                                             bow,
                                             n_char,
                                             batch_start_idx_shifted,
                                             batch_size,
                                             num_batch)
        nb_accu.append(nb_accu_avg)
        hmm_accu.append(hmm_accu_avg)
    print '===================================================================='
    print '\tNaive Bayes Classifier         : %f'%np.mean(nb_accu)
    print '\tHidden Markov Model Classifier : %f'%max(hmm_accu)
    print '===================================================================='

    return n_char, np.mean(nb_accu), max(hmm_accu)


def evaluate(lines, vocab, bow, n_char, batch_start_idx, batch_size, num_batch):
    '''
    evaluation of Naive Bayse and HMM, caculate average accuracy of a five
    fold cross-validation
    '''
    hmm_accu_all = []
    nb_accu_all = []
    # cross validation, 5-fold
    for fold in range(5):
        train_idx = batch_start_idx[0 : fold * num_batch] + \
                   batch_start_idx[(fold + 1) * num_batch:]
        test_idx = batch_start_idx[fold * num_batch :
                                   (fold + 1) * num_batch]
        # training Naive Bayes
        prob_c, prob_c_after_c = train_c_to_c(lines,
                                              bow,
                                              train_idx,
                                              batch_size,
                                              n_char)
        prob_w_by_c = train_w_by_c(lines,
                                   bow ,
                                   train_idx,
                                   batch_size,
                                   n_char)

        # test Naive Bayes
        prob_l_by_c = test_l_by_c(prob_w_by_c, bow, test_idx, batch_size)

        # test
        for t in test_idx:
            # test HMM
            hmm_speakers = get_speakers(lines,
                                        prob_l_by_c,
                                        prob_c_after_c,
                                        prob_c,
                                        t,
                                        batch_size)
            # test naive bayes
            nb_speakers = [max(prob_l_by_c[l].iteritems(),
                               key=operator.itemgetter(1))[0]
                           for l in range(t, t + batch_size)]
            # evaluate accuracy
            real_speakers = [lines[l][1]
                             for l in range(t, t + batch_size)]
            hmm_accu = 0.0
            nb_accu = 0.0
            for i in range(batch_size):
                if real_speakers[i] == hmm_speakers[i]:
                    hmm_accu += 1
                if real_speakers[i] == nb_speakers[i]:
                    nb_accu += 1
            hmm_accu = hmm_accu/float(batch_size)
            nb_accu = nb_accu/float(batch_size)
            print 'real speakers :\t\t\t' + str(real_speakers)
            print 'nb predicted speakers : \t' + str(nb_speakers) + \
                  '\taccu =:\t' + str(nb_accu)
            print 'hmm predicted speakers :\t' + str(hmm_speakers) + \
                  '\taccu =:\t' + str(hmm_accu)
            hmm_accu_all.append(hmm_accu)
            nb_accu_all.append(nb_accu)

    nb_accu_avg = np.mean(nb_accu_all)
    hmm_accu_avg = np.mean(hmm_accu_all)
    print '--------------------------------------------------------------------'
    print 'Accuracy of classifing %d characters'%n_char + \
          '\n\taverage naive bayes : %f'%nb_accu_avg + \
          '\n\taverage hmm         :%f'%hmm_accu_avg
    print '--------------------------------------------------------------------'

    return nb_accu_avg, hmm_accu_avg


def test_l_by_c(prob_w_by_c, bow, test_idx, batch_size):
    '''
    calculate the probability of each line spoken by each character.
    '''
    prob_l_by_c = {}

    for t in test_idx:
        for l in range(t, t + batch_size):
            prob_l_by_c[l] = {}
            for c in range(len(prob_w_by_c[0])):
                prob = 0
                for w in range(len(bow[l])):
                    prob += bow[l][w] * prob_w_by_c[w][c]

                prob_l_by_c[l][c] = prob
    return prob_l_by_c


def train_w_by_c(lines, bow, train_idx, batch_size, n_char):
    '''
    calculate probability of each word spoken by each character
    '''
    smooth_w_by_c = 1.0
    prob_w_by_c = [[smooth_w_by_c for i in range(n_char)]
                    for j in range(len(bow[0]))]
    # count
    for t in train_idx:
        for l in range(t, t + batch_size):
            # add bow of line l to prob_w_by_c
            curr_speaker = lines[l][1]
            for w in range(len(bow[0])):
                prob_w_by_c[w][curr_speaker] += bow[l][w]
    # normalization
    for w in range(len(prob_w_by_c)):
        sum_prob_w = sum(prob_w_by_c[w])
        prob_w_by_c[w] = [math.log(float(i)/float(sum_prob_w))
                          for i in prob_w_by_c[w]]

    return prob_w_by_c


def train_c_to_c(lines, bow, train_idx, batch_size, n_char):
    '''
    calculate transition probability matrix between each pair of character
    and probability word spoken by character through training set
    '''
    smooth_c = 0.5
    smooth_c_after_c = 0.5
    prob_c = [smooth_c for i in range(n_char)] # prob of a character speaks
    prob_c_after_c = [[smooth_c_after_c for i in range(n_char)]
                       for j in range(n_char)]
    # count
    for t in train_idx:
        for l in range(t, t + batch_size):
            curr_speaker = lines[l][1]
            prob_c[curr_speaker] += 1
            if l > 0:
                prev_speaker = lines[l-1][1]
                prob_c_after_c[prev_speaker][curr_speaker] += 1
    # normalization and logarithm
    sum_prob_c = sum(prob_c)
    prob_c = [math.log(float(i)/float(sum_prob_c)) for i in prob_c]
    for i in range(n_char):
        sum_prob_c = sum(prob_c_after_c[i])
        for j in range(n_char):
            prob_c_after_c[i][j] = math.log(float(prob_c_after_c[i][j]) /
                                            float(sum_prob_c))

    # print prob_c
    # print prob_c_after_c
    return prob_c, prob_c_after_c



def get_speakers(lines,
                 prob_l_by_c,
                 prob_c_after_c,
                 prob_c,
                 test_idx,
                 batch_size):
    """
    This is the Viterbi algorithm used for decoding speakers for a
    sequence of lines.
    Decode size is batch_size

    Input:
        prob_l_by_c: a dict of dict, first key is line, second is character

    Output:
        hmm_speakers: a list of speaker ids corresponding to the dialog
     """
    c_num = len(prob_c_after_c)

    # v is a matrix of size [line_num][chracter_num],
    # where v[j][i] stands for the maximum probability of some sort of
    # speaker tagging till the j-th sentence in the dialog,
    # where j-th sentence is spoken by the i-th character
    v = [[0 for i in range(c_num)] for j in range(batch_size +1)]
    # print 'len(v) = :\t'+str(len(v))

    # ptr is a back pointer that can help us trace back who the speaker is
    # ptr[j][i] stands for the person of the (j-1)-th sentence
    # given that the j-th sentence is spoken by the i-th character
    ptr = [[0 for i in range(c_num)] for j in range(batch_size +1)]

    # initialization
    if test_idx > 0:
        prev_speaker = lines[test_idx - 1][1]
    for i in range(c_num):
        v[0][i] = 0
        # v[1][i] = prob_c[i] + prob_l_by_c[test_idx][i]
        if test_idx > 0:
            v[1][i] = prob_c_after_c[i][prev_speaker] + prob_l_by_c[test_idx][i]
        else:
            v[1][i] = prob_c[i] + prob_l_by_c[test_idx][i]

    # iteration:
    for l in range(2, batch_size + 1):
        for i in range(c_num):
            max_prob = -float('inf')
            for k in range(c_num):
                prob = (v[l-1][k] +           # the prob of last speaker is k
                        prob_c_after_c[k][i]) # the prob of i speak after k
                if prob > max_prob:
                    ptr[l][i] = k
                    max_prob = prob
            v[l][i] = prob_l_by_c[test_idx + l - 1][i] + \
                      max_prob # l start with 1 ...

    # termination
    hmm_speakers = []
    max_prob = -float('inf')
    for i in range(c_num):
        if v[batch_size][i] > max_prob:
            max_prob = v[batch_size][i]
            last_speaker = i
    hmm_speakers.insert(0, last_speaker)

    for l in range(0, batch_size-1):
        x = ptr[batch_size-l][hmm_speakers[0]]
        hmm_speakers.insert(0, x)
    return hmm_speakers



# if __name__ == '__main__':
#     if (len(sys.argv) > 1):
#         movie_idx = int(sys.argv[1])
#     else:
#         movie_idx = 0
#
#     main(movie_idx)
