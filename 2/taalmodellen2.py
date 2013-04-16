"""
Assignment 2
Eszter Fodor (5873320), Sharon Gieske (6167667) & Jeroen Rooijmans

To run code: python taalmodellen2.py austen.txt n m
"""

import sys
import struct
import random
import time
import math
import re
import operator
from operator import itemgetter


def readfile(filename):
	"""
	Read text from file
	"""

	f = open(filename, "r+")

	# Create unigram list
	for line in f.readlines():
		for word in line.split():
			unigram.append(word)
	f.close()

	
def createTuple(list,index,n):
	z = tuple(list[max(index-n+1,0):index+1])
	return ('START',)*(n-len(z)) + z		
	
def create_ngrams(seq, n):
	"""
	Create ngrams, parameter = length of word sequences
	"""
	dict = {}
	for i in range(-1,len(seq)):
		t = createTuple(seq, i, n)
		if t in dict:
			dict[t] += 1
		else:
			dict[t] = 1
	return dict
	
	
def getWordSequence(sentences):
	seq = []
	for sentence in sentences:
		if sentence != "":
			seq.extend(sentence.split())
			seq.append('STOP')
	return seq
	

def print_sorted(m):
	"""
	Print most frequent ngrams and value, parameter = number of frequent sequences
	"""
	for i in sorted(ngram_dict, key=ngram_dict.get, reverse=True):
		# number of prefered frequent sequences showed, then break
		if m <= 0:
			break;
		else:
			print("%d: %s" % (ngram_dict[i], i))
			m = m -1

	
def getMHighest(dict, m):
	freqs = []
	total = 0
	for (key,value) in sorted(dict.items(),key=itemgetter(1),reverse=True):
		total += value
		freqs.append((key,value))
	return (freqs[:m],total)
	
	
def loadFile(filename,split):
	file = open(filename,'r')
	buffer = file.read()
	filtered_buffer = re.sub(r"\n[\n]+","\n\n",buffer)
	return filtered_buffer.split(split)
			

def main(argv):
	"""
	Program entry point.
	Arguments: filename, number for ngram, number of frequent sequences
	"""

	if len(argv) == 5:
		n = int(argv[1])
		
		# Load corpus
		corpus = loadFile(argv[2],'\n\n')
		corpus_seq = getWordSequence(corpus)
		
		# get ngrams
		ngram = create_ngrams(corpus_seq, n)
		
		(highest,total) = getMHighest(ngram,10)
		print "= 10 most frequent %d-grams =" % n
		for (high,freq) in highest:
			print high,freq
		print "\n"
	else:
		print "Error: Incorrect arguments"

	
	

if __name__ == '__main__':
	sys.exit(main(sys.argv))
