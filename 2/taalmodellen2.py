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

# dictionary for frequency
ngram_dict = dict()
# list for all words
unigram = []

def readfile(filename):
	"""
	Read text from file
	"""

	f = open(filename, "r+")
	#fNew = detectEmptyLine(f)

	# Create unigram list
	for line in f.readlines():
		for word in line.split():
			unigram.append(word)
	f.close()
	
def detectEmptyLine(f):
	
	f = open(f, "r+")
	f2 = open('austen2.txt', 'r+')
	
	for line in f:
		if line == '\n':
			line = '</s>' + '<s>'
		f2.write(line)
	f.close()
	f2.close()
			
	
def create_ngrams(n):
	"""
	Create ngrams, parameter = length of word sequences
	"""
	for i in range(0, len(unigram)-n):
		if i == 0:
			print unigram
		# create ngram tuple
		tuple_ngram = ()
		for j in xrange(n):
			tuple_ngram += (unigram[i+j],)
		# TODO: unigrams: '</s><s>It'

		# ngram already in dictionary
		if (tuple_ngram in ngram_dict):
			ngram_dict[tuple_ngram] += 1;
		# new ngram
		else:
			ngram_dict[tuple_ngram] = 1;		

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

def get_sum():
	"""
	Print out sum of all frequences
	"""
	sum_ngrams = sum(ngram_dict.values())
	print sum_ngrams 
			

def main(argv):
	"""
	Program entry point.
	Arguments: filename, number for ngram, number of frequent sequences
	"""

	if len(argv) == 4:
		detectEmptyLine(argv[1])
		readfile('austen2.txt')
	else:
		print "Error: Incorrect arguments"

	# get ngrams
	n = int(argv[2])
	if n > 1:
		create_ngrams(int(argv[2]))
		create_ngrams(n-1)
	# print frequences
	print_sorted(int(argv[3]))
	# print sum of frequences
	get_sum()
	

if __name__ == '__main__':
	sys.exit(main(sys.argv))
