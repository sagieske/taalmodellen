import sys
import struct
import random
import time
import math
import re
import operator

# dictionary for frequency
ngram_dict = dict()
unigram = []

def readfile(filename):
	"""
	Read from file
	"""

	f = open(filename, "r+")

	# Create unigram list
	for line in f.readlines():
		for word in line.split():
			unigram.append(word)
	f.close()
	
def create_ngrams(ngram):
	"""
	Create ngrams, parameter = n
	"""

	for i in range(0, len(unigram)-ngram):
		# create ngram tuple
		tuple_ngram = ()
		for j in xrange(ngram):
			tuple_ngram += (unigram[i+j],)

		# ngram already in dictionary
		if (tuple_ngram in ngram_dict):
			ngram_dict[tuple_ngram] += 1;
		# new ngram
		else:
			ngram_dict[tuple_ngram] = 1;

	print ngram_dict

def main(argv):
	"""
	Program entry point.
	Arguments: filename, number for ngram
	"""

	if len(argv) == 3:
		readfile(argv[1])
	else:
		print "Error: Incorrect arguments"

	create_ngrams(int(argv[2]))




if __name__ == '__main__':
	sys.exit(main(sys.argv))
