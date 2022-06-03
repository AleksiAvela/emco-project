import pandas as pd
import numpy as np

from utils import data_to_sparse_bow, initialize_data, test
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.preprocessing import normalize

from emco import ExtrapolatedMarkovChainOversampling as EMCO
from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from pydro.src.dro import DistributionalRandomOversampling


'''
This code performs the same tests with the same methods as presented
in the results in the manuscript, but only for one specified class
and without averaging over multiple repetitions.
'''


### Choose one label as the positive class in OvR classification:
category       = 'ipi'
only_headlines = True


### Initialize the data:
train, tr_docs, tr_vocabulary, y_tr, te_docs, y_te = initialize_data(
	   pos_class=category, stopwords=nltk_stopwords.words('english'),
	   only_headlines=only_headlines)	
X_tr = data_to_sparse_bow(tr_docs, tr_vocabulary)
X_te = data_to_sparse_bow(te_docs, tr_vocabulary)

print()
print("\tMinority class ({}) frequency in training data:".format(category), 
	  str(round(100*(sum(y_tr)/len(y_tr)),1))+"%\n")

### Initialize the oversampling methods:
ros   = ROS()
smote = SMOTE(k_neighbors=5)
ada   = ADASYN()
emco  = EMCO(stopwords=nltk_stopwords.words('english'), stem=True,
			 singles=False, min_len=1, gamma='auto')
dro   = DistributionalRandomOversampling(rebalance_ratio=0.5)
train_nwords = np.asarray(X_tr.sum(axis=1)).reshape(-1)
test_nwords  = np.asarray(X_te.sum(axis=1)).reshape(-1)

### General oversampling:
Xros,   yros   = ros.fit_resample(X_tr, y_tr)
Xsmote, ysmote = smote.fit_resample(X_tr, y_tr)
Xada,   yada   = ada.fit_resample(X_tr, y_tr)
### l2-normalize SMOTE and ADASYN samples:
Xsmote = normalize(Xsmote, norm='l2', axis=1)
Xada   = normalize(Xada,   norm='l2', axis=1)

### DRO:
Xdro, ydro = dro.fit_transform(X_tr, np.asarray(y_tr), train_nwords)
Xdro_te    = dro.transform(X_te, test_nwords, samples=1)

### EMCO:
emco.fit(train, y_tr, pos_class=1)
emco_sample = emco.sample(n='full')
emco_docs   = tr_docs.copy()
yemco  = y_tr.copy()
for s in emco_sample:
	emco_docs.append(s.split(' '))
	yemco.append(1)
Xemco = data_to_sparse_bow(emco_docs, tr_vocabulary)

### Imbalance ratio:
IR = sum(y_tr)/len(y_tr)

method, bAcc, TPR, TNR, cost = [], [], [], [], []
for X, y, name in zip([X_tr, Xros, Xsmote, Xada, Xdro, Xemco],
					  [y_tr, yros, ysmote, yada, ydro, yemco],
					  ["Original", "ROS", "SMOTE", "ADASYN", "DRO", "EMCO"]):
	if name == "DRO":
		res = test(X, y, Xdro_te, y_te, r=IR)
	else:
		res = test(X, y, X_te, y_te, r=IR)
	method.append(name)
	bAcc.append(res[0])
	TPR.append(res[1])
	TNR.append(res[2])
	cost.append(res[3])


results = pd.DataFrame({'Method' : method,
						'Bal. Acc.' : bAcc,
						'TPR' : TPR,
						'TNR' : TNR,
						'Total cost' : cost})
print()
print(results)

