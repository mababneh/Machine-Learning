
import pandas as pd
import time
from sklearn.metrics import classification_report
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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
def main():
    DatasetNM = input("Please Enter the name of DataSet (classificationtext1,otherData)\n")
    if DatasetNM == "classificationtext1":
        train = pd.read_csv('train1.csv', header=None)
        T = train.iloc[:, 0: int(train.shape[1])-1].values
        P = train.iloc[:, int(train.shape[1])-1].values
        for i in range(len(P)):
            if P[i] == 0:
                P[i] = -1
        print(P)
        X, Xtest, y, ytest = train_test_split(T, P, random_state=1, test_size=.3)
    if DatasetNM=="otherData":
        # it is used to enter other dataset by usinf the path of dataset file and number of coloumn
        DatasetPath = input("Please Enter the path of DataSet\n")
        train = pd.read_csv(DatasetPath, header=None)
        T = train.iloc[:, 0:int(train.shape[1])-1].values
        P = train.iloc[:, int(train.shape[1])-1].values
        X, Xtest, y, ytest = train_test_split(T, P, random_state=1, test_size=.3)

    def CheckingClassifer(ClassiferName):
        if ClassiferName == "Perceptron":
            # Running the Perceptron Classifier and computing the Test accuracy and CV accuracy
            print("--------------------------------perceptron---------------------------------------------------")
            start_time = time.time()
            eta_P=input("Please Insert the value of learning Rate : \n")
            pip_perce = make_pipeline(StandardScaler(), PCA(n_components=2),
                           Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, max_iter=50, tol=None,
                                      shuffle=True,
                                      verbose=0, eta0=float(eta_P), n_jobs=None, random_state=0, early_stopping=False,
                                      validation_fraction=0.1,
                                      n_iter_no_change=5, class_weight=None,
                                      warm_start=False))
            pip_perce.fit(X, y)
            y_pred = pip_perce.predict(Xtest)
            print("Test Accuracy :  ", metrics.accuracy_score(ytest, y_pred))# Computing the Test accuracy
            print("The Running Time To compute the Test Accuracy :")
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            K3 = StratifiedKFold(n_splits=10, random_state=1, shuffle=True).split(T, P)
            scores1 = []
            for K, (train_index, test_index) in enumerate(K3):
                pip_perce.fit(T[train_index], P[train_index])
                score = pip_perce.score(T[test_index], P[test_index])
                scores1.append(score)
            print(" The accuracy of Cross validation Perceptron : " + str(sum(scores1) / len(scores1)))# Computing The CV Accuracy
            print("The Running Time To compute the CV Accuracy :")
            print("--- %s seconds ---" % (time.time() - start_time))

            print("-----------------------------------------------------------------------------------")
        elif ClassiferName == "SVM":
            # Running the Support Vector Machine (SVM) Classifier and computing the Test accuracy and CV accuracy
            print("--------------------------------SVM---------------------------------------------------")
            start_time = time.time()
            C_SVM=input("Please Inuput the value of C : \n")
            pip_SV = make_pipeline(StandardScaler(), PCA(n_components=2),
                           LinearSVC(random_state=0, tol=1e-5,C=float(C_SVM)))
            pip_SV.fit(X, y)
            y_pred = pip_SV.predict(Xtest)
            print("Test Accuracy:  ", metrics.accuracy_score(ytest, y_pred))# Computing The test accuracy
            print("The Running Time To compute the Test Accuracy :")
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            K3 = StratifiedKFold(n_splits=10, random_state=1, shuffle=True).split(T, P)
            scores2 = []
            for K, (train_index, test_index) in enumerate(K3):
                pip_SV.fit(T[train_index], P[train_index])
                score = pip_SV.score(T[test_index], P[test_index])
                scores2.append(score)
            print(" The accuracy of Cross validation SVM :  "+ str(sum(scores2) / len(scores2)))# Computing The CV accuracy
            print("The Running Time To compute the CV Accuracy :")
            print("--- %s seconds ---" % (time.time() - start_time))
            scores = []
        elif ClassiferName == "DecisionTree":
            # Running the Decision Tree algorithm (DT) Classifier and computing the Test accuracy and CV accuracy
            print("--------------------------------DT---------------------------------------------------")
            start_time = time.time()
            Max_depth=input("Please Insert the Depth of the Tree : \n")
            pip_DT = make_pipeline(StandardScaler(), PCA(n_components=2),
                           DecisionTreeClassifier(random_state=0, max_depth=int(Max_depth)))
            pip_DT.fit(X, y)
            y_pred = pip_DT.predict(Xtest)
            print("Test Accuracy :  ", metrics.accuracy_score(ytest, y_pred))# Computing The test accuracy
            print("The Running Time To compute the Test Accuracy :")
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            K0 = StratifiedKFold(n_splits=10, random_state=1, shuffle=True).split(T, P)
            scores3 = []
            for K, (train_index, test_index) in enumerate(K0):
                pip_DT.fit(T[train_index], P[train_index])
                score = pip_DT.score(T[test_index], P[test_index])
                scores3.append(score)
            print(" The accuracy of Cross validation DT :  " + str(sum(scores3) / len(scores3)))# Computing The CV accuracy
            print("The Running Time To compute the CV Accuracy :")
            print("--- %s seconds ---" % (time.time() - start_time))

        elif ClassiferName == "KNN":
            # Running the k-nearest neighbors algorithm (k-NN) Classifier and computing the Test accuracy and CV accuracy
            print("--------------------------------KNN---------------------------------------------------")
            start_time = time.time()
            N_of_neg=input("Please insert the number of neigbours : \n")
            pip_KNN = make_pipeline(StandardScaler(), PCA(n_components=2),
                           KNeighborsClassifier(n_neighbors=int(N_of_neg)))
            pip_KNN.fit(X, y)
            y_pred = pip_KNN.predict(Xtest)
            print("Test Accuracy: ", metrics.accuracy_score(ytest, y_pred))# Computing the Test accuracy
            print("The Running Time To compute the Test Accuracy :")
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            K1 = StratifiedKFold(n_splits=10, random_state=1, shuffle=True).split(T, P)
            scores4 = []
            for K, (train_index, test_index) in enumerate(K1):
                pip_KNN.fit(T[train_index], P[train_index])
                score = pip_KNN.score(T[test_index], P[test_index])
                scores4.append(score)
            print(" The accuracy of Cross validation KNN :  " + str(sum(scores4) / len(scores4)))#Computing the CV Accuracy
            print("--- %s seconds ---" % (time.time() - start_time))

        elif ClassiferName == "LG":
            #Running the Logestic Regression Classifier and computing the Test accuracy and CV accuracy
            print("--------------------------------LG---------------------------------------------------")
            start_time = time.time()
            C_LG=input("Please Insert C value :\n")
            pip_LG=make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=float(C_LG), fit_intercept=True,
                                      intercept_scaling=1, class_weight=None, random_state=None,max_iter=500,solver='liblinear'))
            pip_LG.fit(X,y)
            y_pred=pip_LG.predict(Xtest)
            print("Test Accuracy :", metrics.accuracy_score(ytest, y_pred))#Comuting the test accuracy
            print("The Running Time To compute the Test Accuracy :")
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            kflod = StratifiedKFold(n_splits=10, random_state=1, shuffle=True).split(T, P)
            scores5 = []
            for K, (train_index, test_index) in enumerate(kflod):
                pip_LG.fit(T[train_index], P[train_index])
                score = pip_LG.score(T[test_index], P[test_index])
                scores5.append(score)
            print(" The accuracy of Cross validation LG : " + str(sum(scores5) / len(scores5)))#Computing The CV Accuracy
            print("--- %s seconds ---" % (time.time() - start_time))

            print("----------------------------------------------------------------------")
    clock=True
    def ERRoRCheckingClassifer():
        # this function is used to check the name of classifier is correct or not
        # if the user input 0 mean the program will be terninated
        while clock==True:
            ClassiferName = input("please Enter The ClassiferName {Perceptron,SVM,LG,KNN,DecisionTree}:\n")
            if ((ClassiferName=="Perceptron") or (ClassiferName=="SVM") or (ClassiferName=="KNN") or(ClassiferName=="DecisionTree") or(ClassiferName=="LG") ):
                CheckingClassifer(ClassiferName)
            elif ClassiferName == str(0):
                print("Thank You for Using My program")
                break
            else:
                print("please Enter the name of the Classifier correctly\n")

    ERRoRCheckingClassifer()


if __name__ == "__main__":
    main()