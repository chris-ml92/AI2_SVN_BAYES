import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score as crossVal
from sklearn.model_selection import train_test_split as split_Trainingset

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
Y = dataSet[:,57]  # spam/not spam array

#-------------------------------------------------------------------------
#Dataset with lengths
X = tf_idf(X)



#Dataset applying the cosine similarity
listofNorms = np.sqrt(((X + 1e-128)**2).sum(axis = 1, keepdims = True)) #||d1|| = square root(d1[0]2 + d1[1]2 + ... + d1[n]2)
X_normalized = np.where(listofNorms > 0.0, X / listofNorms, 0.0) # i kernel poi applicano in automatico la cosine similarity. che è (x1/||x1||)*(x2/||x2||)
#Note that the tf-idf functionality in sklearn.feature_extraction.text can produce normalized vectors, in which case cosine_similarity is equivalent to linear_kernel, only slower.)
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
#Linear Kernel set
#-------------------------------------------------------------------------
print("======= LINEAR KERNEL =======\n")

linear_Classifier = SVC(kernel = "linear")
linear_Scores = crossVal(linear_Classifier, X, Y, cv = 10, n_jobs = 12 ) # n of cpu's used. io ho un i7 8700k, cambia in base al tuo setup in caso
MSE_linear = -1* crossVal(linear_Classifier, X, Y, cv = 10, n_jobs = 12, scoring='neg_mean_squared_error')
print("Linear Kernel mean score value is: " + str(linear_Scores.mean())+"\n")
print("Linear Kernel minimum score value is: " + str(linear_Scores.min()) + "\n")
print("Linear Kernel maximum score value is: " + str(linear_Scores.max()) + "\n")
print("Accuracy: %0.2f (+/- %0.3f)\n" % (linear_Scores.mean(), linear_Scores.std()*2))
print("MSE value is: %0.4f\n" %(MSE_linear.mean()))

print("======= END LINEAR KERNEL =======\n\n\n")




#-------------------------------------------------------------------------
#Linear Kernel Angular set
#-------------------------------------------------------------------------
print("======= ANGULAR LINEAR KERNEL =======\n")

linear_Classifier_ang = SVC(kernel = "linear")
linear_Scores_ang = crossVal(linear_Classifier_ang, X_normalized, Y, cv = 10, n_jobs = 12 )
MSE_linear_ang = -1*crossVal(linear_Classifier_ang, X_normalized, Y, cv = 10, n_jobs = 12, scoring='neg_mean_squared_error')
print("ANGULAR Linear Kernel mean score value is: " + str(linear_Scores_ang.mean())+"\n")
print("ANGULAR Linear Kernel minimum score value is: " + str(linear_Scores_ang.min()) + "\n")
print("ANGULAR Linear Kernel maximum score value is: " + str(linear_Scores_ang.max()) + "\n")
print("Accuracy: %0.2f (+/- %0.3f)\n" % (linear_Scores_ang.mean(), linear_Scores_ang.std()*2))
print("MSE value is: %0.4f\n" %(MSE_linear_ang.mean()))

print("======= END ANGULAR LINEAR KERNEL =======\n\n\n")




#-------------------------------------------------------------------------
#RBF Kernel set
#-------------------------------------------------------------------------
print("======= RBF KERNEL =======\n")
rbf_Classifier = SVC(kernel = "rbf", gamma = "auto")
rbf_Scores = crossVal(rbf_Classifier, X, Y, cv = 10, n_jobs = 12)
MSE_rbf = -1* crossVal(rbf_Classifier, X, Y, cv = 10, n_jobs = 12, scoring='neg_mean_squared_error')
print("RBF Kernel mean score value is: " + str(rbf_Scores.mean())+"\n")
print("RBF Kernel minimum score value is: " + str(rbf_Scores.min()) + "\n")
print("RBF Kernel maximum score value is: " + str(rbf_Scores.max()) + "\n")
print("Accuracy: %0.2f (+/- %0.3f)\n" % (rbf_Scores.mean(), rbf_Scores.std()*2))
print("MSE value is: %0.4f\n" %(MSE_rbf.mean()))

print("======= END RBF KERNEL =======\n\n\n")




#-------------------------------------------------------------------------
#ANGULAR RBF Kernel set
#-------------------------------------------------------------------------
print("======= ANGULAR RBF KERNEL =======\n")
rbf_Classifier_Ang = SVC(kernel = "rbf", gamma = "auto", C = 1.0)
rbf_Scores_Ang = crossVal(rbf_Classifier, X_normalized, Y, cv = 10, n_jobs = 12)
MSE_rbf_ang = -1*crossVal(rbf_Classifier_Ang, X_normalized, Y, cv = 10, n_jobs = 12, scoring='neg_mean_squared_error')
print("ANGULAR RBF Kernel mean score value is: " + str(rbf_Scores_Ang.mean())+"\n")
print("ANGULAR RBF Kernel minimum score value is: " + str(rbf_Scores_Ang.min()) + "\n")
print("ANGULAR RBF Kernel maximum score value is: " + str(rbf_Scores_Ang.max()) + "\n")
print("Accuracy: %0.2f (+/- %0.3f)\n" % (rbf_Scores_Ang.mean(), rbf_Scores_Ang.std()*2))
print("MSE value is: %0.4f\n" %(MSE_rbf_ang.mean()))

print("======= ANGULAR END RBF KERNEL =======\n\n\n")




#-------------------------------------------------------------------------
# Polynomial Kernel Set
#-------------------------------------------------------------------------
print("======= POLYNOMIAL KERNEL =======\n")

polynomial_Classifier = SVC(kernel = "poly", degree = 2, gamma = "auto")
polynomial_Scores = crossVal(polynomial_Classifier, X, Y, cv=10, n_jobs = 12)
MSE_poly = -1*crossVal(polynomial_Classifier, X, Y, cv = 10, n_jobs = 12, scoring='neg_mean_squared_error')
print("Polynomial_d2 Kernel mean score value is: " + str(polynomial_Scores.mean())+"\n")
print("Polynomial_d2 Kernel minimum score value is: " + str(polynomial_Scores.min()) + "\n")
print("Polynomial_d2 Kernel maximum score value is: " + str(polynomial_Scores.max()) + "\n")
print("Accuracy: %0.2f (+/- %0.3f)\n" % (polynomial_Scores.mean(), polynomial_Scores.std()*2))
print("MSE value is: %0.4f\n" %(MSE_poly.mean()))

print("======= END POLYNOMIAL KERNEL =======\n\n\n")




#-------------------------------------------------------------------------
# Angular Polynomial Kernel Set
#-------------------------------------------------------------------------
print("======= ANGULAR POLYNOMIAL KERNEL =======\n")

polynomial_Classifier_Ang = SVC(kernel = "poly", degree = 2, gamma = "auto")
polynomial_Scores_Ang = crossVal(polynomial_Classifier, X_normalized, Y, cv=10, n_jobs = 12)
MSE_poly_ang = -1*crossVal(polynomial_Classifier_Ang, X_normalized, Y, cv = 10, n_jobs = 12, scoring='neg_mean_squared_error')

print("ANGULAR Polynomial_d2 Kernel mean score value is: " + str(polynomial_Scores_Ang.mean())+"\n")
print("ANGULAR Polynomial_d2 Kernel minimum score value is: " + str(polynomial_Scores_Ang.min()) + "\n")
print("ANGULAR Polynomial_d2 Kernel maximum score value is: " + str(polynomial_Scores_Ang.max()) + "\n")
print("Accuracy: %0.2f (+/- %0.3f)\n" % (polynomial_Scores_Ang.mean(), polynomial_Scores_Ang.std()*2))
print("MSE value is: %0.4f\n" %(MSE_poly_ang.mean()))

print("======= ANGULAR END POLYNOMIAL KERNEL =======\n\n\n")




#-------------------------------------------------------------------------
# Create Training set and Test Set
#-------------------------------------------------------------------------
X_train, X_test, Y_train, Y_test = split_Trainingset(X,Y,test_size=0.25)
X_Norm_train, X_Norm_test, Y_Norm_train, Y_Norm_test = split_Trainingset(X_normalized,Y,test_size = 0.25)


#-------------------------------------------------------------------------
# test data given training data
#-------------------------------------------------------------------------
def model_test(modelType, train_x, train_y, test_x, test_y):
    model_fit = modelType.fit(train_x,train_y)
    print(model_fit.score(test_x,test_y))
    print("\n\n")
    predicted = model_fit.predict(test_x) #for debug


print("Linear model score is: ")
model_test(linear_Classifier,X_train,Y_train,X_test,Y_test)

print("Angular linear model score is: ")
model_test(linear_Classifier_ang,X_Norm_train,Y_Norm_train,X_Norm_test,Y_Norm_test)


print("RBF Model score is: ")
model_test(rbf_Classifier,X_train,Y_train,X_test,Y_test)

print("Angular RBF Model score is: ")
model_test(rbf_Classifier_Ang,X_Norm_train,Y_Norm_train,X_Norm_test,Y_Norm_test)


print("Polynomial Model score is: ")
model_test(polynomial_Classifier,X_train,Y_train,X_test,Y_test)

print("Angular Polynomial Model score is: ")
model_test(polynomial_Classifier_Ang,X_Norm_train,Y_Norm_train,X_Norm_test,Y_Norm_test)




#Template
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------

