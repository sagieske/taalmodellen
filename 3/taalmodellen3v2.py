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


def addOne(train,test, log):
	probs = {}
	bigram = create_ngrams(train,2)
	unigram = create_ngrams(train,1)
	unigram[("START",)] = len(train)
	#probs = {}
	for sentence in test:
		seq = sentence.split()
		probs[sentence] = calculateSentenceProb(seq,2,unigram,bigram,"ml",0, log)
		print "%e" % probs[sentence]
	return probs





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
			seq.append('START')
			seq.extend(sentence.split())
			seq.append('STOP')

	return seq



def calculateSentenceProbs(sentences, n, n_1gram, ngram, mode):
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
			p = calculateSentenceProb(seq, n, n_1gram, ngram, mode)
			dict_p[sentence] = p
	return dict_p

def calculateSentenceProb(seq, n, n_1gram, ngram, mode ):
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
		
		# add one smoothing
		if (mode == "add1"):
			count = float(ngram.get(tuple_n,0) +1)

			# TODO: History is nog niet goed.... wat hoort dit te zijn?
			history = n_1gram.get(tuple_n1,0)

			# possible events
			eventsize = len(n_1gram) * len(n_1gram)
			#print history
			p *= count / float((history + (eventsize * 1)))

		# NO smoothing
		if (mode == "normal"):
			if (tuple_n in ngram) and (tuple_n1 in n_1gram):
				p *= float(ngram[tuple_n]) / n_1gram[tuple_n1]
			else:	# propability will be zero: break
				return 0
	print p
	if p == 0:
		print mode
		print "KUT"
	return p



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


def main(argv):
	"""
	Program entry point.
	Arguments: filename, number for ngram, corpus, test file
	"""


	"""
	# If correct amount of arguments calculate ngrams etc.
	if len(argv) == 4:
		n = int(argv[1])
		# get ngrams
		ngram = create_ngrams(corpus_seq, n)
		if n > 1:
			n_1gram = create_ngrams(corpus_seq,(n-1))
		else:
			n_1gram = {}

		# Load example1 file
		ex1_sentences = loadFile(argv[3], '\n')

		# Calculate conditional probabilities
		print "= conditional probabilities ="
		probs = calculateConditionalProbs(ex1_sentences,n, n_1gram, ngram)
		for sentence, p in probs.iteritems():
			seq = sentence.split()
			if seq != []:
				print "P(%s|%s) = %s " % ( seq[-1] , seq[:-1], p)		
		print "\n"
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
		#sentence_prob = calculateSentenceProbs(testcorpus, n, n_1gram, ngram, "normal")
		#for sentence, p in sentence_prob.iteritems():
		#	seq = sentence.split()
		#	print "P(%s) = %s " % ( seq, p )
		#print '\n' 
		#(highest,total) = getMHighest(sentence_prob,10)
		#for (sentence,p) in highest:
		#	seq = sentence.split()
		#	print "P(%s) = %s " % ( seq, p )


		# Add one smoothing
		print "= sentence probabilities ADD ONE="
		sentence_prob = calculateSentenceProbs(testcorpus, n, n_1gram, ngram, "add1")
		(highest,total) = getMHighest(sentence_prob,10)
		#for (sentence,p) in highest:
		#	seq = sentence.split()
		#	print "P(%s) = %s " % ( seq, p )

	# Else error	
	else:
		print "Error: Incorrect arguments"


if __name__ == '__main__':
	sys.exit(main(sys.argv))
