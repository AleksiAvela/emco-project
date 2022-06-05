import pandas as pd
import numpy as np
import json
from datetime import datetime

from utils import data_to_sparse_bow, initialize_data, test
from nltk.corpus import reuters
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.preprocessing import normalize

from emco import ExtrapolatedMarkovChainOversampling as EMCO
from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from pydro.src.dro import DistributionalRandomOversampling


'''
This code was used for obtaining the test results presented in the manuscript.
Please note that running this code will take several hours. If you wish to run
tests on just one class, please see the file "example.py" in the folder.

For the source codes and documentations of the other methods, please refer to:
	https://imbalanced-learn.org/stable/index.html
	https://github.com/AlexMoreo/pydro
'''


save_results_as_json = False
json_file            = 'all_results.json'

### Number of times to repeat oversampling & testing for one class. The presented
### results are averages of these repetitions:
N = 5

### Class frecuency limits to aggregate the results:
###    i) HF  : f >= limits[0]
###   ii) LF  : limits[1] <= f < limits[0]
###  iii) VLF : f < limits[1]
limits = [0.05, 0.01]

categories = []
for label in reuters.categories():
	### Include only categories that have higher frequencies than 5 in the training data
	### as SMOTE uses 5 nearest (minority) neighbors for each minority observation:
	if sum(['training' in index for index in reuters.fileids(label)]) > 5:
		categories.append(label)

results = {'minority_freq' : np.zeros(len(categories)),
      'bAcc' : {'Original' : np.zeros(len(categories)),
				'ROS'      : np.zeros(len(categories)),
				'SMOTE'    : np.zeros(len(categories)),
				'ADASYN'   : np.zeros(len(categories)),
				'DRO'      : np.zeros(len(categories)),
				'EMCO'     : np.zeros(len(categories))},
	   'TPR' : {'Original' : np.zeros(len(categories)),
				'ROS'      : np.zeros(len(categories)),
				'SMOTE'    : np.zeros(len(categories)),
				'ADASYN'   : np.zeros(len(categories)),
				'DRO'      : np.zeros(len(categories)),
				'EMCO'     : np.zeros(len(categories))},
	   'TNR' : {'Original' : np.zeros(len(categories)),
				'ROS'      : np.zeros(len(categories)),
				'SMOTE'    : np.zeros(len(categories)),
				'ADASYN'   : np.zeros(len(categories)),
				'DRO'      : np.zeros(len(categories)),
				'EMCO'     : np.zeros(len(categories))},
      'cost' : {'Original' : np.zeros(len(categories)),
				'ROS'      : np.zeros(len(categories)),
				'SMOTE'    : np.zeros(len(categories)),
				'ADASYN'   : np.zeros(len(categories)),
				'DRO'      : np.zeros(len(categories)),
				'EMCO'     : np.zeros(len(categories))}}


for i, category in enumerate(categories):
	
	print("\nCategory:", category, "\t", datetime.now().strftime("%H:%M:%S"), "\n")
	
	res = {'Original' : np.zeros((N,4)),
		   'ROS'      : np.zeros((N,4)),
		   'SMOTE'    : np.zeros((N,4)),
		   'ADASYN'   : np.zeros((N,4)),
		   'DRO'      : np.zeros((N,4)),
		   'EMCO'     : np.zeros((N,4))}
	
	for j in range(N):
		
		if j == 0:
			
			train, tr_docs, tr_vocabulary, y_tr, te_docs, y_te = initialize_data(
				   pos_class=category, stopwords=nltk_stopwords.words('english'))	
			X_tr = data_to_sparse_bow(tr_docs, tr_vocabulary)
			X_te = data_to_sparse_bow(te_docs, tr_vocabulary)
			
			ros   = ROS()
			smote = SMOTE(k_neighbors=5)
			ada   = ADASYN()
			emco  = EMCO(stopwords=nltk_stopwords.words('english'), stem=True,
						 singles=False, min_len=1, gamma='auto')
			dro   = DistributionalRandomOversampling(rebalance_ratio=0.5)
			train_nwords = np.asarray(X_tr.sum(axis=1)).reshape(-1)
			test_nwords  = np.asarray(X_te.sum(axis=1)).reshape(-1)
			### Fitting EMCO doesn't include randomness so do it only once pre each class:
			emco.fit(train, y_tr, pos_class=1) 
		
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
		emco_sample = emco.sample(n='full')
		emco_docs   = tr_docs.copy()
		yemco  = y_tr.copy()
		for s in emco_sample:
			emco_docs.append(s.split(' '))
			yemco.append(1)
		Xemco = data_to_sparse_bow(emco_docs, tr_vocabulary)
		
		### Imbalance ratio:
		IR = sum(y_tr)/len(y_tr)
		
		### Collect the out-of-sample test results:
		for X, y, name in zip([X_tr, Xros, Xsmote, Xada, Xdro, Xemco],
							  [y_tr, yros, ysmote, yada, ydro, yemco],
							  ["Original", "ROS", "SMOTE", "ADASYN", "DRO", "EMCO"]):
			if name == "DRO":
				bAcc, TPR, TNR, total_cost = test(X, y, Xdro_te, y_te, r=IR)
			else:
				bAcc, TPR, TNR, total_cost = test(X, y, X_te, y_te, r=IR)
			res[name][j,0] = bAcc
			res[name][j,1] = TPR
			res[name][j,2] = TNR
			res[name][j,3] = total_cost
	
	results['minority_freq'][i] = IR
	### Store the averages of N repetitions per each category:
	for name in ["Original", "ROS", "SMOTE", "ADASYN", "DRO", "EMCO"]:
		r = np.average(res[name], axis=0)
		results['bAcc'][name][i] = r[0]
		results['TPR'][name][i]  = r[1]
		results['TNR'][name][i]  = r[2]
		results['cost'][name][i] = r[3]


### Print the aggregated test results:		
bacc = pd.DataFrame(results['bAcc'])
tpr  = pd.DataFrame(results['TPR'])
tnr  = pd.DataFrame(results['TNR'])
cost = pd.DataFrame(results['cost'])
bacc.insert(loc=0, column='f', value=results['minority_freq'])
tpr.insert(loc=0, column='f', value=results['minority_freq'])
tnr.insert(loc=0, column='f', value=results['minority_freq'])
cost.insert(loc=0, column='f', value=results['minority_freq'])

results_agg = []
for df in [bacc, tpr, tnr, cost]:
	results_agg.append(
		pd.DataFrame({'HF'  : df.loc[df.f >= limits[0]].mean()[1:],
			      'LF'  : df.loc[df.f < limits[0]].loc[df.f >= limits[1]].mean()[1:],
			      'VLF' : df.loc[df.f < limits[1]].mean()[1:]}).round(3))

print()
print("\t*******************************")
print("\t*** AGGREGATED TEST RESULTS ***")
print("\t*******************************\n")
print("\tClass frequency limits:", str(100*limits[0])+"%,", str(100*limits[1])+"%")
for i, s in enumerate(['Balanced Accuracy',
		       'True Positive Rate',
		       'True Negative Rate',
		       'Misclassification Cost']):
	print("\n\t\t***", s, "***")
	print(results_agg[i])
	

### Save the average results per each category as a json-file:
if save_results_as_json:
	results_json = {'category'  : categories,
                        'frequency' : results['minority_freq'].tolist()}
	for stat in ['bAcc','TPR','TNR','cost']:
		results_json[stat] = {k : v.tolist() for k, v in results[stat].items()}
	with open(json_file, 'w') as fp:
		json.dump(results_json, fp, indent=4)


