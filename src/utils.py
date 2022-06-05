from scipy.sparse import dok_matrix
from nltk.corpus import reuters
from nltk.tokenize import RegexpTokenizer
import snowballstemmer

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer 
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics


def preprocess(data, min_df=1, min_len=1, stopwords=[], sep=None, stem=True):
	
	'''
	data            : array or list of the documents
	min_df, min_len : a word is included in the vocabulary only if its frequency in
			  the data is strictly greater than min_df, and if its length is
			  strictly greater than min_len
	stopwords       : list of stopwords to be removed from the vocabulary
	sep             : symbol separating words; if None is given, the documents are
			  tokenized with RegexpTokenizer(r'[a-zA-Z]+')
	stem            : whether to (snowball) stem the words
	---
	returns         : list of preprocessed documents as lists of lower case tokens
			  and dictionaries of {word : index} and {word : frequency}
	'''
		
	stemmer = snowballstemmer.stemmer('english')
	
	docs       = [] # Preprocessed documents
	infrequent = {} # Words whose frequencies are not (yet) higher than min_df
	vocabulary = {} # Dictionary of words and their indices
	word_count = {} # Dictionary of words and their frequencies
	
	for i in range(len(data)):
		
		if sep:
			text = data[i].split(sep)
		else:
			text = RegexpTokenizer(r'[a-zA-Z]+').tokenize(data[i])
		
		clean_text = []
		if stem:
			for w in text:
				token = stemmer.stemWord(w.lower())
				if len(token) > min_len and token not in stopwords:
					clean_text.append(token)
		else:
			for w in text:
				token = w.lower()
				if len(token) > min_len and token not in stopwords:
					clean_text.append(token)
		
		if len(clean_text) == 0:
			clean_text = ['<empty>']
		
		for w in clean_text:
			if w in vocabulary.keys():
				word_count[w] += 1
			elif min_df == 0:
				vocabulary[w] = len(vocabulary)
				word_count[w] = 1
			elif w in infrequent.keys():
				if infrequent[w] == min_df:
					vocabulary[w] = len(vocabulary)
					word_count[w] = min_df + 1
				else:
					infrequent[w] += 1
			else:
				infrequent[w] = 1
		
		docs.append(clean_text)
	
	return [[w for w in d if w in vocabulary] for d in docs], vocabulary, word_count


def data_to_sparse_bow(documents, vocabulary, tf=True, idf=True, norm='l2'):
	
	'''
	tf   : apply sublinear tf-transformation, i.e., ln(tf)+1 (if tf > 0)
	idf  : enable (smoothed) idf-transformation
	norm : normalize with either l1 or l2
	---
	returns : sparse tf-idf weighted and normalized bow-matrix
	'''
    
	corpus = [' '.join(doc) for doc in documents]
	pipe   = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
					   ('tfidf', TfidfTransformer(norm=norm,
						     use_idf=idf,     
						     sublinear_tf=tf))]).fit(corpus)
	return dok_matrix(pipe.transform(corpus))
	

def initialize_data(pos_class, stopwords=[], only_headlines=True):
	
	'''
	Initialize Reuters-21578 dataset with ModApte test-training split for binary
	classification given the positive class. By default considers only headlines.
	---
	returns:
		train            : unpreprocessed training documents (used by EMCO)
		train_docs       : preprocessed training documents
		train_vocabulary : training vocabulary
		y_train          : binary training labels
		test_docs        : preprocessed test documents (where min_df=0)
		y_test           : binary test labels
	'''
	
	train   = [] # unpreprocessed training documents
	test    = [] # unpreprocessed test documents
	y_train = [] # binary training labels given the category pos_class
	y_test  = [] # binary test labels given the category pos_class
	
	for file in reuters.fileids():
		if file.split('/')[0] == 'test':
			if only_headlines:
				test.append(reuters.raw(file).split('\n')[0])
			else:
				test.append(reuters.raw(file))
			y_test.append(int(pos_class in reuters.categories(file)))
		else:
			if only_headlines:
				train.append(reuters.raw(file).split('\n')[0])
			else:
				train.append(reuters.raw(file))
			y_train.append(int(pos_class in reuters.categories(file)))
	
	train_docs, train_vocabulary, _ = preprocess(
		train, min_df=1, stopwords=stopwords, stem=True)
	test_docs, _, _ = preprocess(test, min_df=0, stem=True)
	
	return train, train_docs, train_vocabulary, y_train, test_docs, y_test
	

def test(X, y, X_test, y_test, r=0.5):
	
	'''
	X      : (oversampled) training data
	y      : (oversampled) binary training labels
	X_test : test data
	y_test : binary test labels
	r      : imbalance ratio that is used for computing the misclassification costs as
		 cost(FP) = 1 and cost(FN) = (1-r)/r. r = 0.5 corresponds to equal costs.
	---
	Performs classification tests with the given data. SVM with a linear kernel is used
	in classification. Returns balanced accuracy, TPR, TNR and total misclassification
	cost, assuming that the ratio between the cost of FN and the cost of FP is inversely 
	proportional to the imbalance ratio (as oversampling has been done for full balance).
	'''
	
	clf = make_pipeline(StandardScaler(with_mean=False), SVC(kernel='linear'))
	clf.fit(X, y)
	predicted = clf.predict(X_test)
	
	costs = [1, (1-r)/r]
	bAcc = metrics.balanced_accuracy_score(y_test, predicted)
	TPR  = metrics.recall_score(y_test, predicted, pos_label=1)
	TNR  = metrics.recall_score(y_test, predicted, pos_label=0)
	cMat = metrics.confusion_matrix(y_test, predicted)
	total_cost = costs[0]*cMat[0,1] + costs[1]*cMat[1,0]
	
	return bAcc, TPR, TNR, total_cost

