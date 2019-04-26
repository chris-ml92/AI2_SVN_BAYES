import numpy as np
from sklearn.model_selection import cross_val_score as crossVal
from sklearn.model_selection import train_test_split as split_Trainingset


def load_data(file):
    read = open(file, "r")
    data = np.loadtxt(read, delimiter=",")
    return data


def model_test(modelType, train_x, train_y, test_x, test_y):
    model_fit = modelType.fit(train_x,train_y)
    print(model_fit.score(test_x,test_y))
    print("\n\n")
    predicted = model_fit.predict(test_x) #for debug





Train_set = "Data/spambase.data"
dataSet = load_data(Train_set)

X = dataSet[:,:54] # skipped last values as Torsello said
Y = dataSet[:,57]  # spam/not spam array
X = X/100
X_train, X_test, Y_train, Y_test = split_Trainingset(X,Y,test_size=0.25)





from sklearn.naive_bayes import GaussianNB as gaussian
gaussian_Classifier = gaussian()
gaussian_Scores_Ang = crossVal(gaussian_Classifier, X, Y, cv=10, n_jobs = 12)
MSE_poly_ang = -1*crossVal(gaussian_Classifier, X, Y, cv = 10, n_jobs = 12, scoring='neg_mean_squared_error')

print("Guassian mean score value is: " + str(gaussian_Scores_Ang.mean())+"\n")
print("Gaussian minimum score value is: " + str(gaussian_Scores_Ang.min()) + "\n")
print("Gaussian maximum score value is: " + str(gaussian_Scores_Ang.max()) + "\n")
print("Accuracy: %0.2f (+/- %0.3f)\n" % (gaussian_Scores_Ang.mean(), gaussian_Scores_Ang.std()*2))
print("MSE value is: %0.4f\n" %(MSE_poly_ang.mean()))
print("Naive Bayes Model score is: ")
model_test(gaussian_Classifier, X_train,Y_train,X_test,Y_test)




