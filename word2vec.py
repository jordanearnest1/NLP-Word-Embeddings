import os,sys,re,csv
import pickle
from collections import Counter, defaultdict
import numpy as np
import scipy
import math
import random
import nltk
import string
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from numba import jit
from matplotlib import pyplot as plt
import csv



#... (1) First load in the data source and tokenize into one-hot vectors.
#... Since one-hot vectors are 0 everywhere except for one index, we only need to know that index.


#... (2) Prepare a negative sampling distribution table to draw negative samples from.   ##what should this output?! the encoding rep or the index?

#... Consistent with the original word2vec paper, this distribution should be exponentiated.


#... (3) Run a training function for a number of epochs to learn the weights of the hidden layer.
#... This training will occur through backpropagation from the context words down to the source word.


#... (4) Re-train the algorithm using different context windows. See what effect this has on your results.


#... (5) Test your model. Compare cosine similarities between learned word vectors.


###########################################################
#################     Global Variables     ################
###########################################################

random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10

vocab_size = 0
hidden_size = 100
uniqueWords = [""]                      #... list of all unique tokens
wordcodes = {}                          #... dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()                  #... how many times each token occurs
samplingTable = []                      #... table to draw negative samples from



##############################################################################################
#################    Load in the data and convert tokens to one-hot indices   ################
##############################################################################################


def loadData(filename):
    global uniqueWords, wordcodes, wordcounts
    override = False
    if override:
        #... for debugging purposes, reloading input file and tokenizing is quite slow
        #...  >> simply reload the completed objects. Instantaneous.
        fullrec = pickle.load(open("w2v_fullrec.p","rb"))
        wordcodes = pickle.load( open("w2v_wordcodes.p","rb"))
        uniqueWords= pickle.load(open("w2v_uniqueWords.p","rb"))
        wordcounts = pickle.load(open("w2v_wordcounts.p","rb"))
        return fullrec


    #... load in the unlabeled data file. You can load in a subset for debugging purposes.
    handle = open(filename, "r", encoding="utf8")
    fullconts =handle.read().split("\n")
    fullconts = [entry.split("\t")[1].replace("<br />", "").replace("\\", "").replace(".", "") for entry in fullconts[1:(len(fullconts)-1)]]


    ##takes everything makes it a single string separated by spaces. outputting a list.

    #... apply simple tokenization (whitespace and lowercase)
    ##joining it in a long string.
    fullconts = [" ".join(fullconts).lower()]


    print ("Generating token stream...")
    #... (TASK) populate fullrec as one-dimension array of all tokens in the order they appear.
    #... ignore stopwords in this process
    #... for simplicity, you may use nltk.word_tokenize() to split fullconts.
    #... keep track of the frequency counts of tokens in origcounts.

    ##nltk is a natural language tool kit. a fam of text analysis tools. has a lot of things trained on corpuses, can suggest spelling, give counts. tokenize split things out into different words, can also use for stemming (get rid of pre- suffixes )

    #stopwords is a function. stop words are words like "i, me, it". getting rid of random stuff before vectorize

    fullrec = []
    min_count = 50  ## fix put min count back to 50
    origcounts = Counter()   ##counter is a python method under collections. you can set the minimum threshold to only return the 5000 most common words, etc.
    stopword_lst = stopwords.words('english')   ##takes stop words from nltk, taking in english parameter and outputs a list of stopwords in the english language.
    # print(fullconts[0][0:100])
    tokenizedwords = nltk.word_tokenize(fullconts[0])  ##fullconts is one long string. so need to index 0, cause there's only one index.
    #print(tokenizedwords[0:10])
    for word in tokenizedwords:
        if word not in stopword_lst:
            fullrec.append(word)
            origcounts[word] +=1   ##it's like a dictionary, but it's fine if the key is new. can call like a dictionary

    ## tokenize has all the words, got rid of all the garbage word. tokenize can do stripping by whitespace, or tokenize by part of speech.

    print ("Performing minimum thresholding..")
    #... (TASK) populate array fullrec_filtered to include terms as-is that appeared at least min_count times   (have a long list, array of words, go through the list and if there are rare words put the word UNK)
    #... replace other terms with <UNK> token.
    #... update frequency count of each token in dict wordcounts where: wordcounts[token] = freq(token)

    fullrec_filtered = [] #... fill in

    for word in fullrec:
        if origcounts[word] >= min_count:
            fullrec_filtered.append(word)
            # if word in word
            wordcounts[word] +=1
        else:
            fullrec_filtered.append("UNK")

    #... after filling in fullrec_filtered, replace the original fullrec with this one.
    fullrec = fullrec_filtered

##fullrec is our giant list of words. unk is when it's a rare word.

    print ("Producing one-hot indicies")
    #... (TASK) sort the unique tokens into array uniqueWords
    #... produce their one-hot indices in dict wordcodes where wordcodes[token] = onehot_index(token)
    #... replace all word tokens in fullrec with their corresponding one-hot indices.

    uniqueWords = list(set(fullrec))  ##it's making each word appear one time. list() turns something into a list. giant ordered list of every unique word.
    #print(uniqueWords[0:100])
    wordcodes = {}   #... fill in   ##soon we're not going to look at the words, just have a list and use the indices of each word in that list. going to loop over full rec, and represent everysingle word as a list of 0 with a 1, in accordance. wordcodes is setting it up so that "wordcodes[word4] = [0, 0, 0, 1, 0]"  each word is a key, and the value is the list of 0's with a 1. wordcodes is the element. give each word a number.

    ## like this if word3 appears:
    ## word1 word2 word3 word4
    ##  0      0     1     0

##going through everything in our sequential list, uniqueWords, and building up its one-hot encoding, and then assigning that encoding as the value for that word in a dict.

##enumerate creates a dictionary, will thus createa dictionary for unique words

    for key, value in enumerate(uniqueWords):
        wordcodes[value] = key

    fullrec = [wordcodes[word]for word in fullrec]

    ##fullrec becomes a list of all the words as they occur, represented as their one-hot index.

##ex: freq. {a:3, b:2, c:1}
#sample : wordcodes = {}
## for k in freq.keys():
## wordcodes[]

       ## if have a list like unique words, can call that list.index(), and call in the thing we're looking for. everything in the length of [0:].

##if you call set() finds unique words in a list

##one-hot is given a given word, you can look it up in the one-hot list and find other words that are part of it's context, and if it's close will return 1, everything else will be 0

    #... close input file handle
    handle.close()


    #... store these objects for later.
    #... for debugging, don't keep re-tokenizing same data in same way.
    #... just reload the already-processed input data with pickles.
    #... NOTE: you have to reload data from scratch if you change the min_count, tokenization or number of input rows

    pickle.dump(fullrec, open("w2v_fullrec.p","wb+"))
    pickle.dump(wordcodes, open("w2v_wordcodes.p","wb+"))
    pickle.dump(uniqueWords, open("w2v_uniqueWords.p","wb+"))  ##establishes vocabular. list of each word in order
    pickle.dump(dict(wordcounts), open("w2v_wordcounts.p","wb+"))

    ##pickle is a file input/output lib. reads objects and de/serializing.
    #... output fullrec should be sequence of tokens, each represented as their one-hot index from wordcodes.

    return fullrec

data = loadData('unlabeled-text.tsv')   ##need to change back to data = loadData('unlabeled-text.tsv')


##############################################################
#################    Compute sigmoid value    ################
##############################################################

@jit(nopython=True)
def sigmoid(x):  #a discerning function.
    return float(1)/(1+np.exp(-x))

##############################################################################################
#################     generate a table of cumulative distribution of words    ################
##############################################################################################


def negativeSampleTable(train_data, uniqueWords, wordcounts, exp_power=0.75):
    global wordcodes
    #... stores the normalizing denominator (count of all tokens, each count raised to exp_power)
    max_exp_count = 0

##we want the max_exp_count to equal the maximum value. counter should return the highest value.???s

    print ("Generating exponentiated count vectors")
    #... (TASK) for each uniqueWords, compute the frequency of that word to the power of exp_power
    #... store results in exp_count_array.
    exp_count_array = [pow(wordcounts[word], exp_power) for word in uniqueWords] ## outcome is a list that follows the oder of unique words. exp_count_array is a ray of numbers. this is called list comprehension.
    max_exp_count = sum(exp_count_array)



##could try
    # print ("Generating exponentiated count vectors")
    # exp_count_array = []
    # for word_place in uniqueWords:
    #   value = wordcounts[word_place]
    #   exp_count_array.append(value**exp_power)
    # max_exp_count = sum(exp_count_array)



    print ("Generating distribution")

#   #... (TASK) compute the normalized probabilities of each term.
#   #... using exp_count_array, normalize each value by the total value max_exp_count so that
#   #... they all add up to 1. Store this corresponding array in prob_dist
    prob_dist = [num / max_exp_count for num in exp_count_array]




###could try
    # prob_dist = [num / max_exp_count for num in exp_count_array]
    # print(len(prob_dist))
    # len_prob = len(prob_dist)



#takes exp_count_array and divides by the sum, max_exp_count, to normalize each one, so they all are between 0-1

    print ("Filling up sampling table")
#   #... (TASK) create a dict of size table_size where each key is a sequential number and its value is a one-hot index
#   #... the number of sequential keys containing the same one-hot index should be proportional to its prob_dist value
#   #... multiplied by table_size. This table should be stored in cumulative_dict.
#   #... we do this for much faster lookup later on when sampling from this table.

    cumulative_dict = {}
    table_size = 1e8
    start = 0
    i = 0
    previousEnd = 0
    # while i < table_size:
    for i in range(0, len(uniqueWords)):
        end = round(start + prob_dist[i] * table_size)
        for j in range(start, end):
            cumulative_dict[j] = wordcodes[uniqueWords[i]]
        start = end


#       currentSegmentLength = prob_dist[x] * table_size  ##x is the index, unique words is in the same order as prob_dist
#       previousEnd += int(currentSegmentLength)
# ##figure out how to update

#       for num in range(previousEnd, previousEnd+ int(currentSegmentLength)):
#           # cumulative_dict[i] = x           ##unsure if it's this or the following line
#           cumulative_dict[i] = wordcodes[uniqueWords[x]]
#           i += 1



#   # filling up sampling table ---> can try this, too
#   cumulative_dict = {}
#   table_size = 1e8
#   ids = 0
#   start = 0
#   ct = 0
#   for ea in prob_dist:
#       num= int(round(ea * table_size)):
#       if num != 0:
#           for i in range(start, start + num):
#               cumulative_dict = ids
#               ct +=1
#           start = num
#           ids += 1
#       else:
#           ids +=1

# cumulative_dict = []
##word codes is a the one hot vector

##the key is sequential numbers from 1 - 100m
# the value is the one-hot encoding.
#the currentSegmentLength is how many times we put it in our table.

    #sequential numbers from 0-100m

##hash generator. a hash is a long string from hex, a special encoding of words. base 16. python stores key:values, in buckets based on hashes. the "woman" key is really 56690ab830282c7420...
#instead of doing control f "woman" through all the data, data bases

    return cumulative_dict



# # #............................................................................
# # #... generate a specific number of negative samples
# # #.................................................................................
    #... (TASK) randomly sample num_samples token indices from samplingTable.
    #... don't allow the chosen token to be context_idx.
    #... append the chosen indices to results
##if the random number is context idx we ignore it.


##should be 11 thousand-ish unique words.

def generateSamples(context_idx, num_samples):
    global samplingTable, uniqueWords, randcounter  ##0 - 10^8, unique words = 0
    results = []
    for n in range(0, num_samples):
        while len(results) != num_samples:
            rand_idx = random.randint(0, len(samplingTable)-1)
            if context_idx != samplingTable[rand_idx]:
                results.append(samplingTable[rand_idx])

    return results


##would be good if i could return two negative sample words for each context word. in a tuple or a mlist. make a new variable to put these into.
#come up with random numbers and append the values to it.




    #... (TASK) implement gradient descent. Find the current context token from context_word_ids
    #... and the associated negative samples from negative_indices. Run gradient descent on both
    #... weight matrices W1 and W2.
    #... compute the total negative log-likelihood and store this in nll_new.

            ##first calculating terms out, updating w2 for each of 3 words
        # next updating w1, happens once per center token
        # next updating error term, happens once per context word.


##updating the w's one row at a time. won't see a lot of change. learning rate is around 5e5. can up learning rate to .1 and see if it changes a lot more.
# context word is a 1 and
        ##negative sampling is the part ##i is a context word in contextwordid
        #updating weights.

@jit(nopython=True)
def performDescent(num_samples, learning_rate, center_token, context_word_ids,W1,W2,negative_indices):
    # sequence chars was generated from the mapped sequence in the core code
    nll_new = 0


    # neg_things = [neg_indices[x:x+2] for x in range (0, len(negative_indices), 2)]  --> per Chris

    # for k in range(0, len(context_word_ids)):
    for i_d in range(0, len(context_word_ids)):   ##looking at each context token. NEED TO GET THE ID FROM THE INDEX STORED IN CONTEXTWORDIDS, Then use that index to get appropriate word from unique word.

        current_neg_samples = negative_indices[(i_d*num_samples):(i_d*num_samples+num_samples)]   ##would give me first two numbers
        pos_id = context_word_ids[i_d]
        neg_id1 = current_neg_samples[0]
        neg_id2 = current_neg_samples[1]
        h = W1[center_token]

        ##starting for first negative word, updates for W2
        j = W2[neg_id1]
        dot_prod = np.dot(j, h)
        sig_n1 = sigmoid(dot_prod)  #collapse down
        output_neg1=  (sig_n1 - 0)
        error_neg1 = output_neg1 * j
        sig_output = (learning_rate) *output_neg1 * h
        j_new = j - sig_output  ##updating
        e_n1 = np.log((sigmoid(np.negative(dot_prod))))
        W2[neg_id1] = j_new

        ##for 2nd negative word for W2
        j = W2[neg_id2]
        dot_prod = np.dot(j, h)
        sig_n2 = sigmoid(dot_prod)
        output_neg2 = (sig_n2 - 0)
        error_neg2 = output_neg2 * j
        sig_output = (learning_rate) *output_neg2 * h
        j_new = j - sig_output
        e_n2 = np.log((sigmoid(np.negative(dot_prod))))
        W2[neg_id2] = j_new  ##updating w has to be the last thing we do!

        ##updating positive for W2
        j = W2[pos_id]
        dot_prod = np.dot(j, h)
        sig_p = sigmoid(dot_prod)
        output_pos = sig_p - 1
        error_pos = output_pos * j
        sig_output = (learning_rate) * output_pos * h
        j_new = j - sig_output
        e_pos = np.negative(np.log((sig_p)))
        W2[pos_id] = j_new

        ##compute positive word error
        ##updating center word for W1
        h_new = h - (learning_rate)*(error_neg1 + error_neg2 + error_pos)
        W1[center_token] = h_new

        nll_intermediate = e_pos - (e_n1 + e_n2)
        nll_new += nll_intermediate
    return [nll_new]

        ##updating error term is a check, don't update weights with it, just a running accumulator function. once we get to a certain amount of error, tells us we can stop updating our weight matrices (at around 65 thousand)

#when take the slice, removing the other columns. saying for the 1 word, and the 100 h's. create an h every time loop over a context word.
# -log(y_predict) - the nll for the neg contextwords.

# taking the sum of nll for both indices capturing neg and pos error to update weight one. continuously adding the negtative log of the pos sigma and the nll. at the context word level. update w with sum of each error. for each center token, updating w1 once, w2 3 times


    ##going through this 3 times. twice for neg, once for the pos. each time we are going to update the word embeddings. but we want to use the old for all three of them.

    #hidden layer is a saved variable of an array of w1, indexed to our context word, to get that row. we know what our word is, let's index to our matrix and pull out that row.

    #turn our contextwords into


    ##also need to find the appropriate rows in W1. and row in w2.
        # store_context_words = []  #will store the 3 context words for i (1 pos and 2 neg)
        # for k in negative_indices[i]:   ##negative_indices is a 1-dimension list. should be 8 numbers in each negative indices

        ##then, have to find the sigmoid of W2, then W1. (page 5, do bottom eq. first,tr then top eq on pg 6, then second from bottom on pg 5)
        # ' means two

        #the sigmoid is the dot product: w2 by h
        #tj is a binary. 0,1

        #when you find the row in w1, that is h
        #when you find the row in w2, becomes h', or vector 2 prime
        #v is the "vector in w for a given index" (w' or w)
        #h .v is associated with a one-hot index

        ##vj t times h is the dot product of the appropraitely indexed row in w2. and the appropriately indexed row of w1.

        ##the learning rate is how much we update. .05 by default. how much you should move sigma. if move too fast, bounces it around oo much. and if too slow, takes too long to do the calculations.


        # will then multiply by h, use np.dot
        ##then going to activate that (the product of the vj' h (transposed using np.dot)) by our sigmoid. then subtract by our tj. if our context word is a positive word, subtract by 1. otherwise, tj is going to be zero.

        # that product is multiply by h, as well as our learning rate
        # then have what our vj prime used to be, subtract it by the full calculation (above).
        #with that product, set that whole thing equal to the new row.
        #at the end of this, we're going to be updating, doing our calculation to vj'new. making the row we pulled from our matrix, updating it and putting it back in.
        #
            # store_context_words.append()
        # for j in mapped_context


#when update w2, need to update the old vector for a gradient

##weight vectors are a string of 0's

           ##looking at the negative samples (from negative indicies) for each context token


       ##context_word_ids is a dummy variable. need to find the real variable where i call this function. called mapped_context. loopign through the.
    #                   negative_indices += generateSamples(q, num_samples).
    #which index referrs to a pos context word and a negative one.
    #after identfying the neg_indices, do the calc.

##see chapter 16, pages 8 and 9
##going to find the indexes for the context words (each word in context window). and negative context words (i in results of negativesamplegen()).
#using the two equations listed on instructions to make updates on w1 and w2


################################################################################################################
####################    learn the weights for the input-hidden and hidden-output matrices     ##################
################################################################################################################


## the job of training is to tweek the weights and the functions in the hidden layer to get the best possible results. can play with the weights. the functions might be set, but respond differently to the different weights

#curW1 = current weight matrix
#curW2 = is second weight matrix
#nll = negative log likelihood   -- treat the sigma is a fo loop. for each token in w1, negative and multiply it by H, pass it to the sigmoid, and the the log of that.


##### REVISE context window, and keep track of saved_W1.data and W2 files. / rename, so you can access all 3 pairs later. ### 


def trainer(curW1 = None, curW2=None):
    global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size,np_randcounter, randcounter
    vocab_size = len(uniqueWords)           #... unique characters
    hidden_size = 100                       #... number of hidden neurons. ## neuron says, given i've got this thing, where do i send stuff? neurons all work together. might be 100 different weights acting at once.
    context_window = [-2,-1,1,2]            #... specifies which context indices are output. Indices relative to target word. Don't include index 0 itself.
    # context_window = [-4,-3,-2,-1]    
    # context_window = [1,2,3,4]


    nll_results = []                        #... keep array of negative log-likelihood after every 1000 iterations
    internum_lst = []

## fixes problem 5 and 6

    #... determine how much of the full sequence we can use while still accommodating the context window
    start_point = int(math.fabs(min(context_window)))  ##fabs is the abs val. abs val of the first thing in the list (going to be -2), which evaluates to 2. because you want to start at index 0, essentially. mapped sequence is our fullrec (the pickle of data will load in later)
    end_point = len(fullsequence)-(max(max(context_window),0))
    mapped_sequence = fullsequence


    #... initialize the weight matrices. W1 is from input->hidden and W2 is from hidden->output.
    ##creating a giant weight matrice. input-> hidden and hidden -> output. you want to take an input and get and output. you can have a set of functions that can go from cat to meow. the functions have a a sequence of functions, each resolves down. a bunch of smaller layers, pushes from one layer to another, hidden is the term for the layer that isn't the input and isn't the output. we need one set of weights to tell it to go to "slot 800" at the hidden layer. and then another set that says "given that we wound up at slot 800, what are we going to produce as output". hideen layer isn't necessarily the same size as the vocab. if the "marble" drops in from input

    ##vocab_size is the len of unique words.
    if curW1==None:

        np_randcounter += 1
        W1 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
        W2 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
    else:
        #... initialized from pre-loaded file
        W1 = curW1
        W2 = curW2


    #... set the training parameters
    epochs = 1   ##training epochs. a complete run of everything. will go through every word in the sequence and as we go we tell each word in the context window to change a bit "hey hungry, we saw you next to cat, you should change a bit". go all the way through. then when do the next epoch.
## Fix back to 5!
    num_samples = 2
    learning_rate = 0.05
    nll = 0
    iternum = 0

    #... Begin actual training
    for j in range(0,epochs):
        print ("Epoch: ", j)
        prevmark = 0

        #... For each epoch, redo the whole sequence...
        for i in range(start_point,end_point):  ##if there are 1000 in the seq, will look at index 2, to 997. if you're too close to the edge and try to go too high/low in the context window, will freak out

            if (float(i)/len(mapped_sequence))>=(prevmark+0.1):
                print ("Progress: ", round(prevmark+0.1,1))  #for me. progress bar
                prevmark += 0.1
            if iternum%10000==0:   ## a counter of iterations. this will be true at 10K, 20K, etc.....
                print ("Negative likelihood: ", nll)    ## nll_results is a list of every 10,000 nll calculations.
                nll_results.append(nll)
                internum_lst.append(iternum)
                nll = 0

            ##nll is going to occur (the step where you print it out and append is going to happen every 10,000)

            #... (TASK) determine which token is our current input. Remember that we're looping through mapped_sequence
            iternum += 1

            center_token = mapped_sequence[i]
            # print(center_token)
            if center_token == "UNK":
                continue

            #... fill in ##need to figure out our index and our list.
            #... (TASK) don't allow the center_token to be <UNK>. move to next iteration if you found <UNK>.


            #... now propagate to each of the context outputs

            mapped_context = [mapped_sequence[i+ctx] for ctx in context_window]   ### passes in a onehot index

            #indexes of the context words. can expect 4.. context window means looking at [-2,-1,1,2]. right now i is going to be the index (a number). the value of whatever mapped_seq[#] becomes mapped_context.
            ## mapped_context is going to be a list of elements that match that, 4 words. mapped context really starts as an empty list, then do the for loop 4 times and append it each time.

            ##mapped context is going to be the one-hot indices of the words two before and two after a given word


            negative_indices = []   ##indexes of the negative sampling for all those context words. expect 8.
            for q in mapped_context:  ##mapped context is just the index of pos context words. q is going to be a onehot indice representing a word in mappedcontext
                negative_indices += generateSamples(q, num_samples)
            #... implement gradient descent
            # print(center_token)
            [nll_new] = performDescent(num_samples, learning_rate, center_token, mapped_context, W1,W2, negative_indices)   ##the mapped_context is the seqence chars that appears in the perform descent

            ##changed from wordcodes[centertoken] to unique words[centertoken]
            #then turn uniqueWords[center_token] to center_token (an index)


            nll += nll_new   ## Used to be -=, DJ discussion said it needs to be +=


        for nll_res in nll_results:
            print (nll_res)

        neg_log_vales = nll_results
        internum_vals = internum_lst
        plt.scatter(internum_vals, neg_log_vales, color = "green")
        plt.show()

    return [W1,W2]


####################################################################################
####################       Load in a previously-saved model.     ###################
####################################################################################

# #... Loaded model's hidden and vocab size must match current model.


def load_model(filename1, filename2):
  handle = open(filename1,"rb")
  W1 = np.load(handle)
  handle.close()
  handle = open(filename2,"rb")
  W2 = np.load(handle)
  handle.close()
  return [W1,W2]

  # def load_model():
  # handle = open("saved_W1.data","rb")
  # W1 = np.load(handle)
  # handle.close()
  # handle = open("saved_W2.data","rb")
  # W2 = np.load(handle)
  # handle.close()
  # return [W1,W2]




####################################################################################
#################### Save the current results to an output file. ###################
####################################################################################
## ... Useful when computation is taking a long time.
# #... so in the word2vec network, there are actually TWO weight matrices that we are keeping track of. One of them represents the embedding
# #... of a one-hot vector to a hidden layer lower-dimensional embedding. The second represents the reversal: the weights that help an embedded
# #... vector predict similarity to a context word.

def save_model(W1,W2):
  handle = open("saved_W1.data","wb+")
  np.save(handle, W1, allow_pickle=False)
  handle.close()

  handle = open("saved_W2.data","wb+")
  np.save(handle, W2, allow_pickle=False)
  handle.close()



####################################################################################
#################### Code to start up the training function. ###################
####################################################################################

word_embeddings = []
proj_embeddings = []
def train_vectors(preload=False):  ##preload might mean starting where it left off. saying if it's preloaded, pass in the model. otherwise none.
    global word_embeddings, proj_embeddings
    if preload:
        [curW1, curW2] = load_model()
    else:
        curW1 = None
        curW2 = None
    [word_embeddings, proj_embeddings] = trainer(curW1,curW2)
    save_model(word_embeddings, proj_embeddings)


# #.................................................................................
# #... for the averaged morphological vector combo, estimate the new form of the target word
# #.................................................................................

##shouldn't expect good, it's overfitted

# def morphology(word_seq):
#   global word_embeddings, proj_embeddings, uniqueWords, wordcodes
#   embeddings = word_embeddings
#   vectors = [word_seq[0], # suffix averaged
#   embeddings[wordcodes[word_seq[1]]]]
#   vector_math = vectors[0]+vectors[1]
#   #... find whichever vector is closest to vector_math
#   #... (TASK) Use the same approach you used in function prediction() to construct a list
#   #... of top 10 most similar words to vector_math. Return this list.





# #.................................................................................
# #... for the triplet (A,B,C) find D such that the analogy A is to B as C is to D is most likely
# #.................................................................................

# def analogy(word_seq):
#   global word_embeddings, proj_embeddings, uniqueWords, wordcodes
#   embeddings = word_embeddings
#   vectors = [embeddings[wordcodes[word_seq[0]]],
#   embeddings[wordcodes[word_seq[1]]],
#   embeddings[wordcodes[word_seq[2]]]]
#   vector_math = -vectors[0] + vectors[1] - vectors[2] # + vectors[3] = 0
#   #... find whichever vector is closest to vector_math
#   #... (TASK) Use the same approach you used in function prediction() to construct a list
#   #... of top 10 most similar words to vector_math. Return this list.




################################################################################
##################  Find top 10 most similar words to a target word  ###########
################################################################################

  #... (TASK) search through all uniqueWords and for each token, compute its similarity to target_word.
  #... you will compute this using the absolute cosine similarity of the word_embeddings for the word pairs.
  #... Note that the cosine() function from scipy.spatial.distance computes a DISTANCE so you need to convert that to a similarity.
  #... return a list of top 10 most similar words in the form of dicts,
  #... each dict having format: {"word":<token_name>, "score":<cosine_similarity>}


def prediction(target_word):
    global word_embeddings, uniqueWords, wordcodes
    # targets = [target_word]
    outputs = []

    outputs= {}

    target_word = wordcodes[target_word]

    for i in range(0, len(uniqueWords)):
        if i == target_word:
            continue
        else:
            prediction = 1 - (cosine(word_embeddings[i], word_embeddings[target_word]))
            outputs[i] = prediction

    top = sorted(outputs, key = outputs.get, reverse = True)

    top_ten = top[0:10]
    print(top_ten)

    top_dict_outputs = []    ##create a new list and fill it with all the unique words given every index we have in the top ten.

    for ind in top_ten:
        diction = {}              ##word_embeddings spits out an array of numbers
        diction["word"] = uniqueWords[ind]
        diction["score"] = outputs[ind]
        top_dict_outputs.append(diction)

    return top_dict_outputs  ##could need to be a tuple

################################################################################
#################          Task 4 - Prediction        ##########################
################################################################################

def task_4_prediction(row):
    global word_embeddings, uniqueWords, wordcodes
    s1_idx = uniqueWords.index(row[1])
    s2_idx = uniqueWords.index(row[2])
    distance = cosine(word_embeddings[s1_idx], word_embeddings[s2_idx])
    word_similarity = 1 - distance
    return [row[0], word_similarity]



################################################################################
###############################  Running Main  ################################
################################################################################

if __name__ == '__main__':
    if len(sys.argv)==2:    # if True:
        filename = sys.argv[1]
        #... load in the file, tokenize it and assign each token an index.
        #... the full sequence of characters is encoded in terms of their one-hot positions
        fullsequence = loadData(filename)   

        print ("Full sequence loaded...")

        #... now generate the negative sampling table
        print ("Total unique words: ", len(uniqueWords))
        print("Preparing negative sampling table")
        samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)


#       #... we've got the word indices and the sampling table. Begin the training.
#       #... NOTE: If you have already trained a model earlier, preload the results (set preload=True) (This would save you a lot of unnecessary time)
#       #... If you just want to load an earlier model and NOT perform further training, comment out the train_vectors() line
#       #... ... and uncomment the load_model() line


### REVISE - put in the file names corresponding to the particular set of weights (e.g., saved_W2.data) #### 
        
        train_vectors(preload=False)  ##this i commented out for problem 7
        # [word_embeddings, proj_embeddings] = load_model("saved_W1.data", "saved_W2.data")





      #... we've got the trained weight matrices. Now we can do some predictions

        targets = ["good", "bad", "scary", "funny"]
        for targ in targets:
            print("Target: ", targ)
            bestpreds= (prediction(targ))
            for pred in bestpreds:
                print (pred["word"],":",pred["score"])
            print ("\n")
                with open("p8_output_1.txt", "w", newline = '') as results_csv:
                    r_csv = csv.writer(results_csv, delimiter = ',')
                    r_csv.writerow(targ, pred["word"], pred["score"])
                    for x in totals:
                        r_csv.writerow([x[0], x[1], x[2]])
                results_csv.close()
                f.close


##### REVISE ### 

        ##when you run with context window, C = [-4;-3;-2;-1], save to "p8_output_1.txt"
        ## C = [1,2,3,4] p8_output_2.txt
        ## C = [-2,-1,1,2], p9_output.txt






################################################################################
##################           Implement part 4          #########################
################################################################################

        rdata = []
        f = open('intrinsic-test_v2.tsv', 'r', encoding = 'utf-8')
        for x in f.readlines()[1:]:
            rdata.append(re.split('\t', x.replace('\n', '')))

        totals = []
        for row in rdata:
            totals.append(task_4_prediction(row))

        with open('intrinsic_predictions_1.csv', 'w', newline = '') as results_csv:
            r_csv = csv.writer(results_csv, delimiter = ',')
            r_csv.writerow(["id", "similarity"])
            for x in totals:
                r_csv.writerow([x[0], x[1]])
        results_csv.close()
        f.close


#       #... try an analogy task. The array should have three entries, A,B,C of the format: A is to B as C is to ?
        # print (analogy(["son", "daughter", "man"]))
        # print (analogy(["thousand", "thousands", "hundred"]))
        # print (analogy(["amusing", "fun", "scary"]))
        # print (analogy(["terrible", "bad", "amazing"]))



#       #... try morphological task. Input is averages of vector combinations that use some morphological change.
#       #... see how well it predicts the expected target word when using word_embeddings vs proj_embeddings in
#       #... the morphology() function.

#       s_suffix = [word_embeddings[wordcodes["stars"]] - word_embeddings[wordcodes["star"]]]
#       others = [["types", "type"],
#       ["ships", "ship"],
#       ["values", "value"],
#       ["walls", "wall"],
#       ["spoilers", "spoiler"]]
#       for rec in others:
#           s_suffix.append(word_embeddings[wordcodes[rec[0]]] - word_embeddings[wordcodes[rec[1]]])
#       s_suffix = np.mean(s_suffix, axis=0)
#       print (morphology([s_suffix, "techniques"]))
#       print (morphology([s_suffix, "sons"]))
#       print (morphology([s_suffix, "secrets"]))



    else:
        print ("Please provide a valid input filename")
        sys.exit()
