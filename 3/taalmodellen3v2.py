import sys
import math
import re
import itertools
from operator import itemgetter

def main(args):
	model = args[1]

	# Load train corpus
	trainC = loadFile(args[2],'\n\n')

	# Load test corpus
	testC = loadFile(args[3],'\n\n')

	if args[4] == "logprobs":
		log = True
	else:
		log = False

	if model == "ML":
		probs = startML(trainC,testC, log)
	elif model == "GT":
		probs = startGT(trainC,testC, log)
	else:
		k = int(model[3:])
		probs = startGTk(trainC,testC,k, log)

	perc = float(probs.values().count(0.0))/len(probs)
	print "**=====**=====**=====**=====**=====**=====**=====**"
	print " Percentage sentences with 0 probability: %.2f" % perc
	print "\n First 5 sentences with 0 probability"
	i = 0
	for s in filter(lambda x: x[1] == 0.0,probs.items())[:5]:
		i += 1
		print "Sentence", i, ":", " ".join(s[0].split())

def startML(trainC,testC, log):
	probs = {}
	bigram = calculateNgram(trainC,2)
	unigram = calculateNgram(trainC,1)
	unigram[("START",)] = len(trainC)
	probs = {}
	for sentence in testC:
		seq = sentence.split()
		probs[sentence] = calculateSentenceProb(seq,2,unigram,bigram,"ml",0, log)
		print "%e" % probs[sentence]
	return probs

def startGT(trainC,testC, log):
	probs = {}
	bigram = calculateNgram(trainC,2)
	(rvalues,sbigram) = smoothGT(bigram)
	specialunigram = calculateSpecialUniGram(sbigram)
	for sentence in testC:
		seq = sentence.split()
		probs[sentence] = calculateSentenceProb(seq,2,specialunigram,sbigram,"gt",rvalues[1]/sum(rvalues.values()), log)
		print "%e" % probs[sentence]
	return probs


def startGTk(trainC,testC,k, log):
	probs = {}
	bigram = calculateNgram(trainC,2)
	(rvalues,sbigram) = smoothGTk(bigram, k)
	specialunigram = calculateSpecialUniGram(sbigram)
	for sentence in testC:
		seq = sentence.split()
		probs[sentence] = calculateSentenceProb(seq,2,specialunigram,sbigram,"gt",rvalues[1]/sum(rvalues.values()), log)
		print "%e" % probs[sentence]
	return probs


def calculateSpecialUniGram(ngram):
	specialunigram = {}
	for key in ngram:
		if key[0] in specialunigram:
			specialunigram[key[0]] += ngram[key]
		else:
			specialunigram[key[0]] = ngram[key]
	return specialunigram

def getMHighest(dict, m):
	freqs = []
	total = 0
	for (key,value) in sorted(dict.items(),key=itemgetter(1),reverse=True):
		total += value
		freqs.append((key,value))
	return (freqs[:m],total)

def calculateNgram(sentences,n):
	dict = {}
	for sentence in sentences:
		seq = getWordSequence(sentence)
		for i in range(0,len(seq)):
			t = createTuple(seq, i, n)
			if t in dict:
				dict[t] += 1
			else:
				dict[t] = 1
	return dict

def createTuple(list,index,n):
	z = tuple(list[max(index-n+1,0):index+1])
	return ('START',)*(n-len(z)) + z

def getWordSequence(sentence):
	seq = []
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

def calculateSentenceProbs(sentences, n, n_1gram, ngram, mode,zeroprob, log):
	for sentence in sentences:
		if sentence != "":
			seq = getWordSequence(sentence)
			p = calculateSentenceProb(seq, n, n_1gram, ngram,mode,zeroprob, log)
			print "P(%s) = %s " % ( seq[:-1], p )


def calculateSentenceProb(seq, n, n_1gram, ngram,mode,zeroprob, log):
	p = 0
	if mode == "ml":
		for i in range(len(seq)):
			t1 = createTuple(seq,i,n)
			t2 = createTuple(seq, i-1, n-1)
			# in case of t1 is known
			if(t1 in ngram):
				if log:
					p += math.log(float(ngram[t1]) / float(n_1gram[t2]))
				else:
					p += float(ngram[t1]) / float(n_1gram[t2])
			else:
				return 0
	elif mode == "gt":
		unistore = {}
		for i in range(len(seq)):
			t1 = createTuple(seq,i,n)
			if(t1 in ngram):
				if t1[0] in unistore:
					p += unistore[t1[0]]
				else:
					unisum = 0
					for key in n_1gram:
						if key == t1[0]:
							unisum += n_1gram[key]
					if ngram[t1] > 0.0:
						if log:
							unistore[t1[0]] = math.log(float(ngram[t1]) / unisum)
						else:
							unistore[t1[0]] = float(ngram[t1]) / unisum
						p += unistore[t1[0]]
			else:
				p += zeroprob
	return math.exp(p)

def getMostLikelyPermutation(seq, n, n_1gram, n_gram):
	maxP = 0.0
	maxSeq = seq
	for s in itertools.permutations(seq):
		p = calculateSentenceProb(s, n, n_1gram, n_gram, 0, false)
		if p > maxP:
			maxP = p
			maxSeq = s
	return maxSeq


def smoothGT(ngram):
	sNgram = {}
	ngram_values = ngram.values()
	N = 24453025
	n0 = N - len(ngram)
	rvalues = {}
	for t in ngram:
		r = ngram[t]
		if r in rvalues:
			sNgram[t] = rvalues[r]
		else:
			if r > 0:
				rvalues[r] = (r+1)*(float(ngram_values.count(r+1))/ngram_values.count(r))
			else:
				rvalues[r] = (r+1)*(float(ngram_values.count(r+1))/n0)
			sNgram[t] = rvalues[r]
	return (rvalues,sNgram)

def smoothGTk(ngram,k):
	sNgram = {}
	ngram_values = ngram.values()
	n1 = float(ngram_values.count(1))
	nK1 = ngram_values.count(k+1)
	N = 24453025
	n0 = N - len(ngram)
	denom = (1-(((k+1)*nK1)/n1))
	rvalues = {}
	for r in range(max(ngram_values)+1):
		nR1 = ngram_values.count(r+1)
		if r == 0:
			rvalues[r] = n1/n0
		elif r > k:
			rvalues[r] = r
		else:
			nR = ngram_values.count(r)
			rvalues[r] = (((r+1)*(float(nR1)/nR))-(r*((k+1)*nK1)/n1))/denom

	for t in ngram:
		r = ngram[t]
		sNgram[t] = rvalues[r]

	return (rvalues,sNgram)


def loadFile(filename,split):
	file = open(filename,'r')
	buffer = file.read()
	filtered_buffer = re.sub(r"\n[\n]+","\n\n",buffer)
	return filtered_buffer.split(split)

if __name__ == '__main__': main(sys.argv)
