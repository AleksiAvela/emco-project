import numpy as np
from random import uniform
from scipy.sparse import dok_matrix
from utils import preprocess


'''
 ************************************************************************************
 * Implementation of the Extrapolated Markov Chain Oversampling (EMCO) method       *
 * for imbalanced text classification developed by A. Avela and P. Ilmonen (2022)   *
 *                                                                                  *
 * EMCO is based on the assumption that the sequential structure of text can be     *
 * partly learned from the majority class. Thus, oversampling with EMCO allows the  *
 * minority feature space to expand, which helps generalizing the minority class.   *
 ************************************************************************************
'''


class ExtrapolatedMarkovChainOversampling:
	
	
	def __init__(self, stopwords=[], sep=None, stem=True,
		           singles=False, min_len=1, gamma='auto'):
		
		'''
		stopwords (list)  : List of stopwords to be removed from the vocabulary
		sep       (str)   : Token separator in data, e.g. ','
				    default: None -> RegexpTokenizer is used
		stem	  (bool)  : Whether to (Snowball) stem the words or not
		singles   (bool)  : If False, words that appear only once in the total 
				    vocabulary are pruned off
		min_len   (int)   : Words that are not longer than min_len are pruned off
				    (see preprocess() in utils.py)
		gamma	  (float) : Discounting parameter for transitions in majority documents
				    default: gamma = minority_freq
		'''
		
		self.stopwords = stopwords
		self.sep       = sep
		self.stem      = stem
		self.singles   = singles
		self.min_len   = min_len
		self.gamma     = gamma
		
		
	def __preprocess_data(self):
						
		min_docs, min_vocabulary, min_counts = preprocess(
			self.min_data, min_df=0, min_len=self.min_len, stopwords=self.stopwords, 
			sep=self.sep, stem=self.stem)
		maj_docs, maj_vocabulary, maj_counts = preprocess(
			self.maj_data, min_df=0, min_len=self.min_len, stopwords=self.stopwords, 
			sep=self.sep, stem=self.stem)
		
		if not self.singles:
			### Identify words that appear only once in the training set:
			inv_min = {}
			for k, v in min_counts.items():
				inv_min[v] = inv_min.get(v, []) + [k]
			inv_maj = {}
			for k, v in maj_counts.items():
				inv_maj[v] = inv_maj.get(v, []) + [k]
			min_ones = inv_min[1]
			maj_ones = inv_maj[1]
			### Delete single words from the other vocabulary and from the
			### minority class word distribution:
			for i, (ones, voc, other_voc) in enumerate(
					zip([min_ones, maj_ones],
					    [min_vocabulary, maj_vocabulary],
					    [maj_vocabulary, min_vocabulary])):
				for token in ones:
					if token not in other_voc:
						del voc[token]
						if i == 0:
							del min_counts[token]
		
		### Arrange vocabulary as [ min_vocabulary, maj-only_vocabulary, <STOP> ]
		words = list(min_vocabulary.keys()).copy()
		for t in maj_vocabulary:
			if t not in min_vocabulary:
				words.append(t)	
		vocabulary = dict([[word, i] for i, word in enumerate(words)])
		vocabulary['<STOP>'] = len(vocabulary)
		
		self.min_docs = [] # Preprocessed minority documents
		self.maj_docs = [] # Preprocessed majority documents
		
		for doc in min_docs:
			clean_doc = []
			for token in doc:
				if token in vocabulary:
					clean_doc.append(token)
			if len(clean_doc) > 0:
				self.min_docs.append(clean_doc)
				
		for doc in maj_docs:
			clean_doc = []
			for token in doc:
				if token in vocabulary:
					clean_doc.append(token)
			if len(clean_doc) > 0:
				self.maj_docs.append(clean_doc)
		
		self.min_vocabulary = min_vocabulary          # Minority vocabulary
		self.vocabulary	    = vocabulary              # Total vocabulary
		self.distinct_words = list(vocabulary.keys()) # List of words in vocabulary
		
		### Minority document length distribution
		self.length_distribution = [len(doc) for doc in self.min_docs]
		
		### If not given, set the value of discounting parameter gamma:
		if self.gamma == 'auto':
			minority_f = len(self.min_docs)
			majority_f = len(self.maj_docs)
			self.gamma = minority_f / (minority_f + majority_f)
		
		### Word distribution in the minority documents:
		self.min_dist = np.array(list(min_counts.values()))
		
			
	def __add(self, first, second, value=1):
		
		### Add given value to the transition count from first word to second word
		### as well as to the row sum corresponding to the first word:
		try:
			self.P[self.vocabulary[first], self.vocabulary[second]] += value
			self.row_sums[self.vocabulary[first]] += value
		except KeyError:
			print("KeyError")
		

	def __fit_transition_probabilities(self):
		
		'''
		vocabulary = ( {min_voc}, {maj-only_voc}, <STOP> )
		P : Unnormalized Markov probability matrix, dim : [ vocabulary x vocabulary ]
		row_sums : The row sums are used when sampling words, see __draw_next()
		'''
		
		### Initialize P matrix and row sum vector:
		self.P = dok_matrix(
			(len(self.distinct_words), len(self.distinct_words)), dtype=np.float32)
		self.row_sums = np.zeros(len(self.distinct_words))
				
		### Minority transitions:
		for doc in self.min_docs:
			self.__add('<STOP>', doc[0])
			for i in range(len(doc)-1):
				self.__add(doc[i], doc[i+1])
			self.__add(doc[-1], '<STOP>')
		
		### Discounted majority transitions (from common words to any words):
		for doc in self.maj_docs:
			for i in range(len(doc)-1):
				if doc[i] in self.min_vocabulary:
					self.__add(doc[i], doc[i+1], value=self.gamma)
		
		### Transitions from majority-only based on the minority word distribution:
		self.P[len(self.min_vocabulary):-1, :len(self.min_vocabulary)] = self.min_dist
		
		### Ensure zero probability back to word itself:
		for i in range(len(self.min_vocabulary)):
			self.P[i,i] = 0
		
		
	def __draw_next(self, current):
		
		### Draw a uniformly distributed random number from [0, row_sum[current]]
		u = uniform(0, self.row_sums[self.vocabulary[current]])
		s = 0
		for idx, weight in self.P[self.vocabulary[current],:].items():
			s += weight
			if s >= u:
				break
		return self.distinct_words[idx[1]]
		
		
	def __chain(self, seed, length):
		
		'''
		Generate a synthetic document as a Markov chain based on the estimated
		transition probability matrix P.
		seed   : initial token in chain; if none is given, the chain will begin
			 with a <STOP> token (that is not included in the returned chain)
		length : length of the generated chain; if none is given, the length will
			 be drawn from the minority document length distribution
		'''
		
		if seed not in self.distinct_words:
			seed = '<STOP>'	
		current = seed
		chain   = [current]
		
		if type(length) == int:
			L = length
		else:
			L = self.length_distribution[int(uniform(0, len(self.length_distribution)))]
		
		i = 0
		while i < L:
			next_word = self.__draw_next(current)
			if next_word != '<STOP>':
				i += 1
				chain.append(next_word)
			current = next_word
			
		while '<STOP>' in chain:
			chain.remove('<STOP>')
		
		return ' '.join(chain)
		
		
	def fit(self, data, y, pos_class=1):
		
		'''
		Preprocess the data (by parameters given in __init__) and estimate
		the transition probability matrix used in oversampling
		---
		data      : list of (unpreprocessed) documents
		y         : list of (binary) labels
		pos_class : positive (minority) class in data
		'''
		
		min_data = np.array([doc for doc,label in zip(data,y) if label==pos_class])
		maj_data = np.array([doc for doc,label in zip(data,y) if label!=pos_class])
		
		self.min_data = min_data
		self.maj_data = maj_data
		
		self.__preprocess_data()
		self.__fit_transition_probabilities()
					
			
	def sample(self, seed='', n=1, length='auto'):
		
		'''
		Generate a synthetic sample using the estimated transition probabilities
		---
		seed   : initial token in chains; if no seed is given each chain will start
			 from <STOP> token (which is not included in the returned documents)
		n      : sample size; if 'full' is given samples for full balance
		length : length of the synthetic documents; if 'auto' is given draws lengths
			 of the documents from the minority document length distribution
		'''
		
		if n == 'full':
			n = len(self.maj_docs) - len(self.min_docs)
		
		if n == 1:
			return self.__chain(seed, length)
		else:
			documents = ["" for _ in range(n)]
			for i in range(n):
				documents[i] = self.__chain(seed, length)
			return documents
	
