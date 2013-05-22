"""
EDIT (by Eszter): Gonna add comments to understand the code
	- run: python taalmodellen4.py training.pos test.pos bla.pos (options)
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

	# Load training corpus
	print '** Loading train corpus'
	(trainWords, trainTags) = loadFile(args[1], False)
	
	# Load test corpus
	print '** Loading test corpus'
	(testWords, testTags) = loadFile(args[2], short)
	
	# Calculate unigrams of training words
	print '** Calculating unigram'
	unigram = calculateNgram(trainWords, 1)
	
	# Calculate trigrams of training POS tags
	print '** Calculating language model'
	languageModel = calculateNgram(trainTags, 3) 
	
	# Calculate the model for words and tags
	print '** Calculating task model (This may take a while)'
	( taskModel, tagStats, wordTags) = calculateTaskModel(trainWords, trainTags, unigram)
	
	# If smoothing is enabled, smooth probabilities
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
	
	# Calculate the most probable POS tag for every word within the test corpus
	# using the Viterbi algorithm and the calculated language and taks models
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
	Parameters: sequence, tags, output file name
	Output: words with their tags
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



def calculateNgram(sentences,n):
	"""
	Create ngrams (adjusted from taalmodellen3v2.py)

	"""
	dict = {}
	for sentence in sentences:
		sentence.reverse() 
		for i in range(0,len(sentence)):
			t = createTuple(seq, i, n)
			dict[t] = dict.get(t,0) +1
		sentence.reverse()
	return dict



def calculateTaskModel(wordsequences, tagsequences, unigram):
	"""
	Calculate probabilities for words and their tags
	"""
	tagStats = {}
	wordTags = {}
	taskmodel = {}
	tagUnigram = calculateNgram(tagsequences, 1)
	aStore = {}
	wordseqlen = len(wordsequences)
	print "*** Processing %d sentences" % wordseqlen
	for i in range(0, wordseqlen):
		for j in range(0, len(wordsequences[i])):
			# Get word and tag
			word = wordsequences[i][j]
			tag = tagsequences[i][j]
			# Add possible tag to corresponding word
			if word in wordTags:
				if tag not in wordTags[word]:
					wordTags[word].append(tag)
			# Add word + tag to list			
			else:
				wordTags[word] = [tag]

			# Task Model
			if (word, tag) not in taskmodel:
				if (word, tag) in aStore:
					a = aStore[(word, tag)]
				else:
					if unigram[(word,)] == 1:
						if tag in tagStats:
							(singleton, total) = tagStats[tag]
							tagStats[tag] = (singleton+1, total+1)
						# add tag to tagStats
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
			# Find word & tag in sequences
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
	(from taalmodellen3v2.py)
	"""
	seq = []
	if sentence != "":
		seq.extend(sentence.split())
		seq.append('STOP')
	return seq


# VOLGENS MIJ WORDT DIT NIET GEBRUIKT..
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
	"""
	Smooth the taks model
	"""
	sTaskModel = {}
	# Smooth occcurances taskmodel
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
	Good-Turing Smoothing with k value (from taalmodellen3v2.py)
	"""
	# smoothed ngrams and rvalues
	sNgram = {}
	rvalues = {}
	values = ngram.values()

	# calculation of constant number of events
	n1 = float(values.count(1))
	# TODO: eventsize = 24453025 -> delete
	eventsize = 24453025
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
	# List for storing sequences
	tagsequences = []
	wordsequences = []
	# Set regular expression object X/Y
	pairs = re.compile(r"(\S+)/(\w[a-zA-Z0-9\$]*) ")
	# Delete empty lines
	filtered_buffer = re.sub(r"\n"," ",buffer)
	# Split end of sentences (./. and =====etc)
	paragraphs = filtered_buffer.split("======================================")
	for p in paragraphs:
		paragraph_parts = p.split("./.")
		for p2 in paragraph_parts:
			tagsequence = []
			wordsequence = []
			# Find and append all X/Y paris
			for (word,tag) in pairs.findall(" "+p2+" "):
				tagsequence.append(tag)
				wordsequence.append(word)
			# Add end of sequence signs
			if wordsequence != []:
				if not short or len(wordsequence) < 16:
					# TODO: Why double stops?
					wordsequence.append('STOP')
					wordsequence.append('STOP')
					tagsequence.append('STOP')
					tagsequence.append('STOP')
					tagsequences.append(tagsequence)
					wordsequences.append(wordsequence)

	return (wordsequences, tagsequences)

if __name__ == '__main__': main(sys.argv)
