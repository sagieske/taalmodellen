import sys
import re
from operator import itemgetter

""" 
Problem statement:
how to deal with the unigram of start
"""


def main(args):
	n = int(args[1])

	# Load corpus
	corpus = loadFile(args[2],'\n\n')
	corpus_seq = getWordSequence(corpus)

	# Calculate n-gram and (n-1)-gram
	ngram = calculateNgram(corpus_seq,n)
	if n > 1:
		n_1gram = calculateNgram(corpus_seq,(n-1))
	else:
		n_1gram = {}

    # Calculate and print 10 most fequent (n-1)-grams
	(highest,total) = getMHighest(n_1gram,10)
	print "= 10 most frequent (%d-1)-grams =" % n
	for (high,freq) in highest:
		print high,freq
	print "\n"
    
    
	# Calculate 10 most frequent n-grams
	(highest,total) = getMHighest(ngram,10)
	print "= 10 most frequent %d-grams =" % n
	for (high,freq) in highest:
		print high,freq
	print "\n"
	
	# Load example1 file
	ex1_sentences = loadFile(args[3], '\n')

	# Calculate conditional probabilities
	print "= conditional probabilities ="
	calculateConditionalProbs(ex1_sentences,n, n_1gram, ngram)
	print "\n"

	# Load example2 file
	ex2_sentences = loadFile(args[4], '\n')

	print "= sentence probabilities ="
	calculateSentenceProbs(ex2_sentences, n, n_1gram, ngram) 


def getMHighest(dict, m):
	freqs = []
	total = 0
	for (key,value) in sorted(dict.items(),key=itemgetter(1),reverse=True):
		total += value
		freqs.append((key,value))
	return (freqs[:m],total)

def calculateNgram(seq,n):
	dict = {}
	for i in range(-1,len(seq)):
		t = createTuple(seq, i, n)
		if t in dict:
			dict[t] += 1
		else:
			dict[t] = 1
	return dict

def createTuple(list,index,n):
	z = tuple(list[max(index-n+1,0):index+1])
	return ('START',)*(n-len(z)) + z

def getWordSequence(sentences):
	seq = []
	for sentence in sentences:
		if sentence != "":
			seq.extend(sentence.split())
			seq.append('STOP')
	return seq

def calculateConditionalProbs(sentences, n,  n_1gram, ngram):
	for sentence in sentences:
		if sentence != "":
			seq = sentence.split()
			p = float(ngram[createTuple(seq,len(seq)-1,n)]) / n_1gram[createTuple(seq, len(seq)-2, n-1)]
	 		print "P(%s|%s) = %s " % ( seq[-1] , seq[:-1], p)

def calculateSentenceProbs(sentences, n, n_1gram, ngram):
	for sentence in sentences:
		if sentence != "":
			seq = sentence.split()
			p = calculateSentenceProb(seq, n, n_1gram, ngram)
			print "P(%s) = %s " % ( seq, p )

def calculateSentenceProb(seq, n, n_1gram, ngram):
	p = 0.0
	for i in range(len(seq)):
		p *= float(ngram[createTuple(seq,i,n)]) / n_1gram[createTuple(seq, i-1, n-1)]
	return p
	
def loadFile(filename,split):
	file = open(filename,'r')
	buffer = file.read()
	filtered_buffer = re.sub(r"\n[\n]+","\n\n",buffer)
	return filtered_buffer.split(split)

if __name__ == '__main__': main(sys.argv)

