"""
Assignment 2
Eszter Fodor (5873320), Sharon Gieske (6167667) & Jeroen Rooijmans

To run code: python taalmodellen2.py n austen.txt ngrams.txt sentences.txt
"""

import sys
import struct
import random
import time
import math
import re
import operator
from operator import itemgetter

"""
Load corpus
"""
def loadFile(filename,split):
	file = open(filename,'r')
	buffer = file.read()
	filtered_buffer = re.sub(r"\n[\n]+","\n\n",buffer)
	return filtered_buffer.split(split)


"""
Add start symbol
"""	
def createTuple(list,index,n):
	z = tuple(list[max(index-n+1,0):index+1])
	return ('START',)*(n-len(z)) + z		
	
	
"""
Create ngrams, parameter = length of word sequences
"""
def create_ngrams(seq, n):
	dict = {}
	for i in range(-1,len(seq)):
		t = createTuple(seq, i, n)
		if t in dict:
			dict[t] += 1
		else:
			dict[t] = 1
	return dict
	
"""
Get sequencies
"""
def getWordSequence(sentences):
	seq = []
	for sentence in sentences:
		if sentence != "":
			seq.extend(sentence.split())
			seq.append('STOP')
	return seq
	

"""
Get the m most highest frequencies
"""
def getMHighest(dict, m):
	freqs = []
	total = 0
	for (key,value) in sorted(dict.items(),key=itemgetter(1),reverse=True):
		total += value
		freqs.append((key,value))
	return (freqs[:m],total)
	
			
"""
Program entry point.
Arguments: filename, number for ngram, corpus, additional file 1, additional file 2
"""
def main(argv):
	
	# If correct amount of arguments calculate ngrams etc.
	if len(argv) == 5:
		n = int(argv[1])
		
		# Load corpus
		corpus = loadFile(argv[2],'\n\n')
		corpus_seq = getWordSequence(corpus)
		
		# get ngrams
		ngram = create_ngrams(corpus_seq, n)
		if n > 1:
			n_1gram = create_ngrams(corpus_seq,(n-1))
		else:
			n_1gram = {}
			
		# Calculate and print 10 most frequent (n-1)-grams
		(highest,total) = getMHighest(n_1gram,10)
		print "= 10 most frequent (%d-1)-grams =" % n
		for (high,freq) in highest:
			print high,freq
		print "\n"
		
		# Calculate and print 10 most frequent ngrams
		(highest,total) = getMHighest(ngram,10)
		print "= 10 most frequent %d-grams =" % n
		for (high,freq) in highest:
			print high,freq
		print "\n"
		
	# Else error	
	else:
		print "Error: Incorrect arguments"


if __name__ == '__main__':
	sys.exit(main(sys.argv))
