import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score as crossVal


Train_set = "Data/spambase.data"


def load_data(file):
	read = open(file, "r")
	data = np.loadtxt(read, delimiter=",")
	return data

def tf_idf(tf):

	nDocuments = tf.shape[0]
	#calculate idf
	idf = np.log10(nDocuments/(tf != 0).sum(0))
	#calculate tf_idf (also I have normalized the probabilities)
	tfidf = (tf/100)*idf
	return tfidf



dataSet = load_data(Train_set)

X = dataSet[:,:54] # skipped last values as Torsello said
Y = dataSet[:,57] # spam/not spam array

X = tf_idf(X)

#-------------------------------------------------------------------------
#Linear Kernel set
#-------------------------------------------------------------------------
print("======= LINEAR KERNEL =======\n")

linear_Classifier = SVC(kernel = "linear")
linear_Scores = crossVal(linear_Classifier, X, Y, cv = 10, n_jobs = 12 ) # n of cpu's used. io ho un i7 8700k, cambia in base al tuo setup in caso

print("Linear Kernel minimum score value is: " + str(linear_Scores.min()) + "\n")
print("Linear Kernel maximum score value is: " + str(linear_Scores.max()) + "\n")

print("======= END LINEAR KERNEL =======\n\n\n")
#-------------------------------------------------------------------------
#RBF Kernel set
#-------------------------------------------------------------------------
print("======= RBF KERNEL =======\n")
rbf_Classifier = SVC(kernel = "rbf", gamma = "auto")
rbf_Scores = crossVal(rbf_Classifier, X, Y, cv = 10, n_jobs = 12)


print("RBF Kernel minimum score value is: " + str(rbf_Scores.min()) + "\n")
print("RBF Kernel maximum score value is: " + str(rbf_Scores.max()) + "\n")

print("")

print("======= END RBF KERNEL =======\n\n\n")
#-------------------------------------------------------------------------
# Polynomial Kernel Set
#-------------------------------------------------------------------------
print("======= POLYNOMIAL KERNEL =======\n")

polynomial_Classifier = SVC(kernel = "poly", degree = 2, gamma = "auto")
polynomial_Scores = crossVal(polynomial_Classifier, X, Y, cv=10, n_jobs = 12)


print("Polynomial_d2 Kernel minimum score value is: " + str(polynomial_Scores.min()) + "\n")
print("Polynomial_d2 Kernel maximum score value is: " + str(polynomial_Scores.max()) + "\n")

print("======= END POLYNOMIAL KERNEL =======\n\n\n")


#Template
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------