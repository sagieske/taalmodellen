"""
EDIT (by Eszter): Gonna add comments to understand the code
	- run: python taalmodellen4.py training.pos test.pos bla.pos
	- started running at ~16:35 -> 17:10 still running -> 17:45 finished:
	
eszter@eszter-laptop /media/DATA/AI/taalmodellen/4 $ python taalmodellen4.py training.pos test.pos bla.pos
![Smoothing enabled]
![Writing output to bla.pos]
** Loading train corpus
** Loading test corpus
** Calculating unigram
** Calculating language model
** Calculating task model (This may take a while)
*** Processing 42916 sentences
** Smooth language model
** Smooth task model
** Calculate special bigram
** Start Tagging
-----------------------
Total words: 49443
Total words tagged: 48103
Precision: 88.894664%
Recall: 86.485448%

"""

import sys
import math
import re
import itertools
from operator import itemgetter


def main(args):
	"""
	Program entry point
	Arguments: training corpus, test corpus, output file, options
	"""
	debug = "--debug" in args
	short =  "--short" in args
	smooth = "--no-smoothing" not in args

	if debug:
		print '![Debug mode on]'
	if short:
		print '![Only short sentences]'
	if smooth:
		print '![Smoothing enabled]'
	else:
		print '![Smoothing disabled]'
	
	outputFileName = args[3]
	print '![Writing output to %s]' % outputFileName
	outputFile = open(outputFileName,'w')

	# Load train corpus
	print '** Loading train corpus'
	(trainWords, trainTags) = loadFile(args[1], False)
	print '** Loading test corpus'
	(testWords, testTags) = loadFile(args[2], short)
	
	
	print '** Calculating unigram'
	unigram = calculateNgram(trainWords, 1)
	print '** Calculating language model'
	languageModel = calculateNgram(trainTags, 3) # 2nd-order Markov Model
	print '** Calculating task model (This may take a while)'
	( taskModel, tagStats, wordTags) = calculateTaskModel(trainWords, trainTags, unigram)
	if smooth:
		print '** Smooth language model'
		(rvalues, sLanguageModel) = smoothGTk(languageModel, 4)
		print '** Smooth task model'
		sTaskModel = smoothTask(taskModel, tagStats)
		print '** Calculate special bigram'
		sBigram = calculateSpecialBiGram(sLanguageModel)
	else:
		sLanguageModel = languageModel
		sTaskModel = taskModel
		sBigram = calculateNgram(trainTags, 2)
	
	print '** Start Tagging'
	total = 0
	correct = 0
	unknown = 0
	for s in range(len(testWords)):
		seq = testWords[s]
		(prob, tags) = viterbi(seq, sLanguageModel, sBigram, sTaskModel, wordTags, debug)
		outputTaggedSentence(seq, tags, outputFile)	
		total += len(seq)-2
		for t in range(1, len(tags)-1):
			if tags[t] == "UNKNOWN":
				unknown += 1
			if tags[t] == testTags[s][t-1]:
				correct += 1
	outputFile.close()
	if total > 0:
		recall = float(correct)/total
		precision = float(correct)/(total-unknown)
		print '-----------------------'
		print 'Total words: %d' % total
		print 'Total words tagged: %d' % (total-unknown)
		print 'Precision: %f%%' % (precision*100)
		print 'Recall: %f%%' % (recall*100)

def outputTaggedSentence(seq, tags, outputFile):
	"""
	Write tagged sequences to the output file
	"""
	for s in range(len(seq)-2):
		word = seq[s]
		tag = tags[s]
		str = "%s/%s " % (word, tag)
		outputFile.write(str)
	outputFile.write("\n")

def viterbi(seq, sLanguageModel, sBigram, sTaskModel, wordTags, debug):
	"""
	Implementation of the Viterbi algorithm
	Calculates emission and transition probabilities
	"""
	V = []
	for w in range(len(seq)):
		word = seq[w]
		emissions = {}
		if word == "STOP":
			emissions['STOP'] = 1
		else:
			if word not in wordTags:
				emissions['UNKNOWN'] = 1
			else:
				for tag in wordTags[word]:
					emissions[tag] = sTaskModel[(word, tag)]
		V.append(emissions)
	
	transitions = {}
	for i in range(len(V)-1):
		for e1 in V[i]:
			if i == 0:
				transitions[("START",e1)] = 1
			else:
				for e2 in V[i-1]:
					for e3 in V[i+1]:
						try:	
							transitions[(e2,e1)] = sLanguageModel[(e3,e1,e2)]/sBigram[(e1,e2)]
						except KeyError:
							if debug:
								print '----------------KeyError on 55-------------------'
								print 'KeyError while calculating transition probability'
								print 'transitions[(e2,e1)] : ','transitions[(%s,%s)]' % (e2,e1)
								print 'sLanguageModel[(e3,e1,e2)] : ' ,'sLanguageModel[(%s,%s,%s)]' % (e3,e1,e2)
								print 'sBigram[(e1,e2)] : ' ,'sBigram[(%s,%s)]' % (e1,e2)
								print 'V:'
								print V
								print '-------------------------------------------------'
							transitions[(e2,e1)] = 0
	(prob, route) = calculateOptimalRoute(V, transitions, len(seq)-2, debug)
	return (prob, route)
	
def calculateOptimalRoute(V,transitions, location, debug):
	"""
	Calculate most probable route based on transition and emisson probabilities
	"""
	if location == -1:
		return (1,["START"])
	else:
		(prevProb, trellis) = calculateOptimalRoute(V, transitions, location-1, debug)
		try:
			(prob, tag) = max([(transitions[(trellis[len(trellis)-1],tag)] * V[location][tag], tag) for tag in V[location]])
		except Exception:
			if debug:
				print '-------------Exception on 73-----------------'
				print 'Exception while calculating optimal route'
				print 'transitions[(trellis[len(trellis)-1],tag)] : ','transition[(%s, %s)]' % (trellis[len(trellis)-1],tag) 
				print '---------------------------------------------'
			prob = 1
			tag = "UNKNOWN"
		trellis.append(tag)
		return (prevProb*prob, trellis)
		
def calculateSpecialBiGram(ngram):
	"""
	Calculate bigrams for usage in 2nd-order Markov Model
	"""
	specialbigram = {}
	for key in ngram:
		if (key[0],key[1]) in specialbigram:
			specialbigram[(key[0],key[1])] += ngram[key]
		else:
			specialbigram[(key[0],key[1])] = ngram[key]
	return specialbigram

def calculateSpecialUniGram(ngram):
	"""
	!!! Not used in this program !!!
	"""
	specialunigram = {}
	for key in ngram:
		if key[0] in specialunigram:
			specialunigram[key[0]] += ngram[key]
		else:
			specialunigram[key[0]] = ngram[key]
	return specialunigram

def getMHighest(dict, m):
	"""
	!!! Not used in this program !!!
	"""
	freqs = []
	total = 0
	for (key,value) in sorted(dict.items(),key=itemgetter(1),reverse=True):
		total += value
		freqs.append((key,value))
	return (freqs[:m],total)

def calculateNgram(sentences,n):
	"""
	Create n grams
	"""
	dict = {}
	for sentence in sentences:
		sentence.reverse()
		for i in range(0,len(sentence)):
			t = createTuple(sentence, i, n)
			if t in dict:
				dict[t] += 1
			else:
				dict[t] = 1
		sentence.reverse()
	return dict

def calculateTaskModel(wordsequences, tagsequences, unigram):
	tagStats = {}
	wordTags = {}
	taskmodel = {}
	tagUnigram = calculateNgram(tagsequences, 1)
	aStore = {}
	wordseqlen = len(wordsequences)
	print "*** Processing %d sentences" % wordseqlen
	for i in range(0, wordseqlen):
		for j in range(0, len(wordsequences[i])):
			word = wordsequences[i][j]
			tag = tagsequences[i][j]
			if word in wordTags:
				if tag not in wordTags[word]:
					wordTags[word].append(tag)
			else:
				wordTags[word] = [tag]
			if (word, tag) not in taskmodel:
				if (word, tag) in aStore:
					a = aStore[(word, tag)]
				else:
					if unigram[(word,)] == 1:
						if tag in tagStats:
							(singleton, total) = tagStats[tag]
							tagStats[tag] = (singleton+1, total+1)
						else:
							tagStats[tag] = (1,1)
					else:
						if tag in tagStats:
							(singleton, total) = tagStats[tag]
							tagStats[tag] = (singleton, total+1)
						else:
							tagStats[tag] = (0,1)
					a = countWordWithTagInSequences(word,tag,wordsequences, tagsequences)
				b = tagUnigram[(tag,)]
				taskmodel[(word, tag)] = float(a) / b
	return (taskmodel, tagStats, wordTags)

def countTagInSequences(tag, sequences):
	"""
	Count appearences of a particular tag
	"""
	count = 0
	for sequence in sequences:
		for t in sequence:
			if t == tag:
				count += 1
	return count

def countWordWithTagInSequences(word, tag, wordsequences, tagsequences):
	"""
	Count appearence of a particular word with a particular tag
	"""
	count = 0
	for i in range(0, len(wordsequences)):
		for j in range(0, len(wordsequences[i])):
			if wordsequences[i][j] == word and tagsequences[i][j] == tag:
				count += 1
	return count

def createTuple(list,index,n):
	"""
	Create tuples and add start symbol
	"""	
	z = tuple(list[max(index-n+1,0):index+1])
	return ('START',)*(n-len(z)) + z

def getWordSequence(sentence):
	"""
	Get sequencies with start and stop symbols
	Argument: corpus of sentences/paragraphs
	Return: sequence array with start and stop symbols
	"""
	seq = []
	if sentence != "":
		seq.extend(sentence.split())
		seq.append('STOP')
	return seq

def calculateConditionalProbs(sentences, n,  n_1gram, ngram):
	"""
	!!! Not used in this program !!!
	"""
	for sentence in sentences:
		if sentence != "":
			seq = sentence.split()
			p = float(ngram[createTuple(seq,len(seq)-1,n)]) / n_1gram[createTuple(seq, len(seq)-2, n-1)]
	 		print "P(%s|%s) = %s " % ( seq[-1] , seq[:-1], p)

def calculateSentenceProbs(sentences, n, n_1gram, ngram, mode,zeroprob, log):
	"""
	!!! Not used in this program !!!
	"""
	for sentence in sentences:
		if sentence != "":
			seq = getWordSequence(sentence)
			p = calculateSentenceProb(seq, n, n_1gram, ngram,mode,zeroprob, log)
			print "P(%s) = %s " % ( seq[:-1], p )


def calculateSentenceProb(seq, n, n_1gram, ngram,mode,zeroprob, log):
	"""
	!!! Not used in this program !!!
	"""
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
	"""
	!!! Not used in this program !!!
	"""
	maxP = 0.0
	maxSeq = seq
	for s in itertools.permutations(seq):
		p = calculateSentenceProb(s, n, n_1gram, n_gram, 0, false)
		if p > maxP:
			maxP = p
			maxSeq = s
	return maxSeq


def smoothGT(ngram):
	"""
	Good-Turing Smoothing
	"""
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

def smoothTask(taskmodel, tagStats):
	sTaskModel = {}
	for (word, tag) in taskmodel:
		if taskmodel[(word, tag)] == 0:
			(n1,N) = tagStats[tag]
			sTaskModel[(word,tag)] = 0.5*(n1/N)
		elif taskmodel[(word,tag)] == 1:
			sTaskModel[(word,tag)] = 0.5
		else:	
			sTaskModel[(word,tag)] = taskmodel[(word,tag)]
	return sTaskModel

def smoothGTk(ngram,k):
	"""
	Good-Turing Smoothing with k value
	"""
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

def loadFile(filename, short):
	"""
	Load file
	"""
	file = open(filename,'r')
	buffer = file.read()
	return parseWallStreet(buffer, short)

def parseWallStreet(buffer, short):
	"""
	Parse module for the corpora
	"""
	tagsequences = []
	wordsequences = []
	tags = re.compile(r"(\S+)/(\w[a-zA-Z0-9\$]*) ")
	filtered_buffer = re.sub(r"\n"," ",buffer)
	paragraphs = filtered_buffer.split("======================================")
	for p in paragraphs:
		paragraph_parts = p.split("./.")
		for p2 in paragraph_parts:
			tagsequence = []
			wordsequence = []
			for (word,tag) in tags.findall(" "+p2+" "):
				tagsequence.append(tag)
				wordsequence.append(word)
			if wordsequence != []:
				if not short or len(wordsequence) < 16:
					wordsequence.append('STOP')
					wordsequence.append('STOP')
					tagsequence.append('STOP')
					tagsequence.append('STOP')
					tagsequences.append(tagsequence)
					wordsequences.append(wordsequence)

	return (wordsequences, tagsequences)

if __name__ == '__main__': main(sys.argv)
