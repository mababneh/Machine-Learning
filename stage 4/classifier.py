from scratch_2 import Perceptron2
from scratch_1 import AdalineSGD
from scratch_3 import AdalineGD

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
def main ():
    DatasetNM=input("Please Enter the name of DataSet (classificationtext1)\n")
    if DatasetNM=="classificationtext1":
        train = pd.read_csv('/Users/Mohammed/Desktop/traindata_1.csv', header=None)
        trainraws=[]
        i=0
        for i in range (50):
            trainraws.append(i)

        X = train.iloc[0:2862, trainraws].values
        #X_normalized = preprocessing.normalize(X, norm='l2')
        y0 = train.iloc[0:2862, 50].values
        y = np.where(y0 ==0, 1, -1)
        print (y)
        ytest1=y
        print("--------------------------------------------------------------------------------------------------")
        test = pd.read_csv('/Users/Mohammed/Desktop/testdata_1.csv', header=None)
        Xtest = test.iloc[0:1083,trainraws].values
        #X_scaled1 = preprocessing.normalize(Xtest, norm='l2')
        ytest0 = test.iloc[0:1083, 50].values
        ytest = np.where(ytest0 == 0, +1, -1)

        y_actu = pd.Series(ytest, name='Actual')

    clock=True
    def CheckingClassifer(ClassiferName):

        if ClassiferName=="All Classifiers":

            pop= Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, max_iter=50, tol=None, shuffle=True,
                            verbose=0, eta0=0.01, n_jobs=None, random_state=0,early_stopping=False, validation_fraction=0.1,
                            n_iter_no_change=5, class_weight=None,
                            warm_start=False)
            pop.fit(X,y)
            print("The Error in each iteration")
            #print(df)
            pop.predict(Xtest)
            #print(dd)
            print (pop.score(Xtest,ytest))

            print("*************************SVM******************************************")
            pip = LinearSVC(random_state=0, tol=1e-5)
            df = pip.fit(X, y)
            print("The Error in each iteration")
            print(df)
            dd = pip.predict(Xtest)
            print(dd)
            print(pip.score(Xtest, ytest))
            print("**************************TreeDescion*******************************************")
            clf = DecisionTreeClassifier(random_state=0,max_depth=15)
            df = clf.fit(X, y)
            print("The Error in each iteration")
            print(df)
            dd = clf.predict(Xtest)
            print(dd)
            print(clf.score(Xtest, ytest))
            print("**************************KNN*******************************************")
            model = KNeighborsClassifier(n_neighbors=10)
            nr = model.fit(X, y)
            print("The Error in each iteration")
            print(nr)
            nrd = model.predict(Xtest)
            print(nrd)
            print("Accuracy:", metrics.accuracy_score(ytest, nrd))
            print("**************************LogisticRegression*******************************************")
            model2=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                      intercept_scaling=1, class_weight=None, random_state=None,max_iter=500,solver='liblinear')
            nr1 = model2.fit(X, y)
            print("The Error in each iteration")
            print(nr1)
            nrd1 = model2.predict(Xtest)
            print(nrd1)
            print("Accuracy:", metrics.accuracy_score(ytest, nrd))


    def ERRoRCheckingClassifer():
        while clock==True:
            ClassiferName = input("please Enter The ClassiferName {All Classifiers}:\n")
            if ((ClassiferName=="All Classifiers")):
                CheckingClassifer(ClassiferName)
            elif ClassiferName == str(0):
                print("Thank You for Using My program")
                break
            else:
                print("please Enter the name of the file correctly\n")

    ERRoRCheckingClassifer()
if __name__ == "__main__":
    main()