"""
Assignment 3
Eszter Fodor (5873320), Sharon Gieske (6167667) & Jeroen Rooijmans

To run code: python taalmodellen3v2.py n austen.txt sentences.txt
"""

import sys
import time
import math
import re
import operator
from operator import itemgetter
import itertools
from collections import Counter
import time






"""
=========== Code from Assignment 2 ============
Permutations and Conditional probs for ngrams are deleted
"""


def loadFile(filename,split):
	"""
	Load corpus and split text with parameter split.
	Arguments: filename (string), split (string)
	"""
	file = open(filename,'r')
	buffer = file.read()

	# substitute all large white areas to one white line
	filtered_buffer = re.sub(r"\n[\n]+","\n\n",buffer)

	# split substituted text by given splitparameter
	return filtered_buffer.split(split)



def createTuple(list,index,n):
	"""
	Create tuples and add start symbol
	"""	
	z = tuple(list[max(index-n+1,0):index+1])
	return ('START',)*(n-len(z)) + z		




def create_ngrams(seq, n):
	"""
	Create ngrams
	Parameters = sequence of words ([str]), length of ngram (int)
	Returns = dictionary of ngrams with frequency
	"""
	dict = {}
	# start at range n-1 due to prepended start symbol. otherwise multiple START symbols will be prepended.

	for i in range(n-1,len(seq)):
		t = createTuple(seq, i, n)
		dict[t] = dict.get(t,0) +1
	return dict

def getWordSequence(sentences):
	"""
	Get sequencies with start and stop symbols
	Argument: corpus of sentences/paragraphs
	Return: sequence array with start and stop symbols
	"""
	seq = []
	for sentence in sentences:
		# if not end ("") add start/stop symbol to sentence
		if sentence != "":
			seq.extend(sentence.split())
			seq.append('STOP')

	return seq

def smoothGTk(ngram, k, eventsize):
	# smoothed ngrams and rvalues
	sNgram = {}
	rvalues = {}
	values = ngram.values()

	# calculation of constant number of events
	n1 = float(values.count(1))
	n0 = eventsize - len(ngram)	
	nk_1 = float(values.count(k+1))
	denom = (((k+1)*nk_1)/n1)
	# for every frequency r seen until k

	for r in range(max(values)+1):
		nr_1 = float(values.count(r+1))
		# use standard Good-Turing Smoothing if r=0
		if r == 0:
			rvalues[r] = n1/n0
		elif r > k:
			rvalues[r] = float(values.count(r))
		else:
			nr = float(values.count(r))
			rvalues[r] = (((r+1)*(float(nr_1)/nr))-(r* denom))/ (1-denom)
	
	
	# create new bigram model with smoothed frequencies
	for t in ngram:
		r = ngram[t]
		sNgram[t] = rvalues[r]

	return (rvalues,sNgram)


def calculateSentenceProbs(sentences, n, n_1gram, ngram, mode, zero_prob):
	"""
	Calculate probability of sentences using function calculateSentenceProb
	Parameters: Sentences ([str]), ngram length (int), n-1gram (dict), ngrams (dict)
	"""
	dict_p = {}
	for sentence in sentences:
		if sentence != "":
			# Split sentence into words
			seq = sentence.split()
			# calculate probability of sentence
			p = calculateSentenceProb(seq, n, n_1gram, ngram, mode, zero_prob)
			dict_p[sentence] = p
	return dict_p

def calculateSentenceProb(seq, n, n_1gram, ngram, mode, zero_prob ):
	"""
	Calculate probability of sentences
	P(w)1, .., w_m) = multiplication from i= 1 to i=m+1 of: P(w_i | w_i-n+1, ..., w_i-1)
		with w_j = START for j<= 0 and w_m+1 = STOP
	Parameters: Sequences ([str]), ngram length (int), n-1gram (dict), ngrams (dict)
	"""
	# start probability of 1 due to multiplication
	p = 1.0
	# calculate probability 
	for i in range(len(seq)):
		# only calculate probability if tuples are in ngram and n_1gram else probability = 0
		if(n > 2 and i <2):
			continue
		# create tuples
		tuple_n = createTuple(seq,i,n)
		tuple_n1 =createTuple(seq,i-1,n-1)	
		
		### Smoothing ###
		# Add one smoothing
		if (mode == "add1"):
			count = float(ngram.get(tuple_n,0) +1)

			# history is number of bigrams
			history = len(ngram)

			# possible events
			eventsize = len(n_1gram) * len(n_1gram)

			p *= count / float((history + (eventsize * 1)))
		
		# Good-Turing smoothing	
		if (mode == "gt"):
			# unigram dictionary
			#totalunigramcounter = {}
			p = 1.0
			for i in range(len(seq)):
				tuple_n = createTuple(seq,i,n)
				# seen bigram
				if (tuple_n in ngram):				
					p *= float(ngram[tuple_n])/n_1gram[tuple_n[0]]
					#print "p: %f" % p
				# not seen bigram 				
				else:
					#print "zero_prob: %f" % zero_prob
					p *= zero_prob
				
		# NO smoothing
		if (mode == "normal"):
			if (tuple_n in ngram) and (tuple_n1 in n_1gram):
				p *= float(ngram[tuple_n]) / n_1gram[tuple_n1]
			else:	# propability will be zero: break
				return 0

	return p

def calculateSpecialUniGram(ngram):
	"""
	Get frequencies unigrams from smoothed bigram
	"""
	# create new unigram dict
	specialunigram = {}
	for key in ngram:
	 	# unigram already in dict, increase counter
		if key[0] in specialunigram:
			specialunigram[key[0]] += ngram[key]
		else:
			specialunigram[key[0]] = ngram[key]

	return specialunigram

def getMHighest(dict, m):
	"""
	Get the m most highest frequencies
	"""
	freqs = []
	total = 0
	for (key,value) in sorted(dict.items(),key=itemgetter(1),reverse=True):
		total += value
		freqs.append((key,value))
	return (freqs[:m],total)

def print_highest(nr, prob_dict):
	(highest,total) = getMHighest(prob_dict,nr)
	for (sentence,p) in highest:
		seq = sentence.split()
		print "P(%s) = %s " % ( seq, p )

def print_zeros(nr,probs):
	percent = float(probs.values().count(0.0))/len(probs)
	print "**=====**=====**=====**=====**=====**=====**=====**"
	print " Percentage sentences with 0 probability: %.2f" % percent
	print "\n First 5 sentences with 0 probability"
	i = 0
	for s in filter(lambda x: x[1] == 0.0,probs.items())[:nr]:
		i += 1
		print "Sentence", i, ":", " ".join(s[0].split())

def main(argv):
	"""
	Program entry point.
	Arguments: trainingcorpus filename, testcorpus filename
	"""

	if len(argv) == 3:
		# Load corpus
		corpus = loadFile(argv[1],'\n\n')
		corpus_seq = getWordSequence(corpus)

		testcorpus = loadFile(argv[2],'\n\n')

		n = 2 # bigram model

		# get ngrams
		ngram = create_ngrams(corpus_seq, n)
		if n > 1:
			n_1gram = create_ngrams(corpus_seq,(n-1))
		else:
			n_1gram = {}


		# Calculate (NORMAL) sentence probabilities
		print "= sentence probabilities NORMAL ="
		sentence_prob_normal = calculateSentenceProbs(testcorpus, n, n_1gram, ngram, "normal", 0)
		#print_highest(10, sentence_prob_normal)
		print_zeros(5 ,sentence_prob_normal)

		# Add one smoothing
		print "= sentence probabilities ADD ONE="
		sentence_prob_add1 = calculateSentenceProbs(testcorpus, n, n_1gram, ngram, "add1", 0)
		
		#print_highest(10, sentence_prob_add1)
		print_zeros(5 ,sentence_prob_add1)

		# Good Turing smoothing
		print "= sentence probabilities Good Turing="
		eventsize = len(n_1gram) * len(n_1gram)
		k = 5
		# created smoothed bigrams
		(rvalues,smoothed_bigram) = smoothGTk(ngram, k, eventsize)
		# create smoothed unigrams
		specialunigram = calculateSpecialUniGram(smoothed_bigram)
		# calculate probability
		zero_probability = float(rvalues[1]/sum(rvalues.values()))
		sentence_prob_gt = calculateSentenceProbs(testcorpus, n, specialunigram, smoothed_bigram, "gt", zero_probability)
		
		# print_highest(10, sentence_prob_gt)
		print_zeros(5 ,sentence_prob_gt)

	# Else error	
	else:
		print "Error: Incorrect arguments"


if __name__ == '__main__':
	sys.exit(main(sys.argv))
