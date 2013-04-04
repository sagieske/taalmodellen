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

	# Create unigram list
	for line in f.readlines():
		for word in line.split():
			unigram.append(word)
	f.close()
	
def create_ngrams(n):
	"""
	Create ngrams, parameter = number of ngrams
	"""

	for i in range(0, len(unigram)-n):
		# create ngram tuple
		tuple_ngram = ()
		for j in xrange(n):
			tuple_ngram += (unigram[i+j],)

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
	Arguments: filename, number for ngram, number of frequent sentences
	"""

	if len(argv) == 4:
		readfile(argv[1])
	else:
		print "Error: Incorrect arguments"

	# get ngrams
	create_ngrams(int(argv[2]))
	# print frequences
	print_sorted(int(argv[3]))
	# print sum of frequences
	get_sum()


if __name__ == '__main__':
	sys.exit(main(sys.argv))
