import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

Train_set = "Data/spambase.data"


def load_data(file):
	read = open(file, "r")
	data = np.loadtxt(read, delimiter=",")
	return data

def tf_idf(email_data):

	nDocuments = email_data.shape[0]
	#calculate idf
	idf = np.log10(nDocuments/(email_data != 0).sum(0))
	#calculate tf_idf (also I have normalized the probabilities)
	tfidf = (email_data/100)*idf
	return tfidf

dataSet = load_data(Train_set)


X = dataSet[:,:54] # skip last values as Torsello said

X = tf_idf(X)

print(X[0]) #works, hopefully well.

