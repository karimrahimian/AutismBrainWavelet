import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import RobustScaler,StandardScaler,QuantileTransformer,MinMaxScaler
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, roc_auc_score
from sklearn.model_selection import  KFold, train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,VotingClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import collections
from sklearn.feature_selection import VarianceThreshold
from sklearn import datasets
from sklearn.decomposition import FastICA
import random
import scipy.io
import numpy as np
import tensorflow as tf
from  sklearn.utils.multiclass import  type_of_target
from scipy import stats
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import StratifiedKFold
from Library.neuroCombat import neuroCombat
import mat73

from collections import Counter
#************************Load initial Files**************************
phenotype = pd.read_excel('Labels.xls')
phenotype=phenotype.to_numpy()
f=Counter(phenotype[:,10])
print (f)
#****************************Classifier******************************
class Pure_Classifier():
    def __init__(self):
        self.evaluationName = ['Precision', 'F1Score', 'Accuracy', 'Recall', 'Matt', 'Auc']
        self.kfold = 5
        self.classifier = []
        pass
    def GenerateAllClassifiers(self):
        ada = AdaBoostClassifier(n_estimators=200, base_estimator=None, learning_rate=0.1, random_state=1)
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        nivebase = GaussianNB()
        dt = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        lr = LogisticRegression(random_state=0)
        svclassifier = SVC(kernel='rbf', degree=5, C=2)
        ridge = RidgeClassifier(alpha=0.5)
        sgdclassifier = SGDClassifier()
        randomforest = RandomForestClassifier(n_estimators=200)  # Train the model on training data
        mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 5), max_iter=1000, alpha=1e-4, solver='sgd', verbose=0,
                            tol=1e-4, random_state=1, learning_rate_init=.08)

        bagging = BaggingClassifier(dt,max_samples=0.5,max_features=0.5)
        voting = VotingClassifier(estimators=[('SVM', svclassifier), ('DT', dt), ('LR', lr)], voting='hard')
        #histogramgradient = HistGradientBoostingClassifier(min_samples_leaf=1,   max_depth = 2,  learning_rate = 1,max_iter = 1)
        myclassifier = [knn,nivebase,dt,lr,svclassifier,randomforest,voting,ridge,mlp,ada,bagging,sgdclassifier]
        return myclassifier

    def GenereateClassifier(self,outclassifier=None):
        ada = AdaBoostClassifier(n_estimators=200, base_estimator=None, learning_rate=0.1, random_state=1)
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        nivebase = GaussianNB()
        dt = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
        lr = LogisticRegression(random_state=0)
        svclassifier = SVC(kernel='rbf', degree=5, C=2)
        ridge = RidgeClassifier(alpha=0.5)
        sgdclassifier = SGDClassifier()
        randomforest = RandomForestClassifier(n_estimators=200)  # Train the model on training data
        mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 5), max_iter=1000, alpha=1e-4, solver='sgd', verbose=0,
                            tol=1e-4, random_state=1, learning_rate_init=.08)

        bagging = BaggingClassifier(dt,max_samples=0.5,max_features=0.5)
        voting = VotingClassifier(estimators=[('SVM', svclassifier), ('DT', dt), ('LR', lr)], voting='hard')
        #histogramgradient = HistGradientBoostingClassifier(min_samples_leaf=1,   max_depth = 2,  learning_rate = 1,max_iter = 1)
        #self.classifier = [knn,nivebase,dt,lr,svclassifier,randomforest,voting,ridge,mlp,ada,bagging,sgdclassifier]
        if (outclassifier!=None):
            self.classifier = [outclassifier]

        #self.classifier = [knn,nivebase,dt,lr,svclassifier,randomforest,voting,ridge]

    def Get_Classifier_Names(self):
        classifier_names = []
        for classifier in self.classifier:
            classifier_names.append(classifier.__class__.__name__)
        return classifier_names
    def multiclass_roc_auc_score(self, y_test, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)
    def __Cross_Validation(self, X, y, classifier, kf):
        Data = X
        evlP = [[0 for x in range(6)] for YY in range(self.kfold)]
        k = 0
        misindex = []
        for train_index, test_index in kf.split(Data,y):
            classifier.fit(Data[train_index], y[train_index])
            y_pred = classifier.predict(Data[test_index])
            y_test = y[test_index]

            evlP[k][0] = (precision_score(y_test, y_pred, average='macro',labels=np.unique(y_pred)))
            evlP[k][1] = (f1_score(y_test, y_pred, average='macro',labels=np.unique(y_pred)))
            evlP[k][2] = (accuracy_score(y_test, y_pred))
            evlP[k][3] = (recall_score(y_test, y_pred, average="weighted"))
            evlP[k][4] = (matthews_corrcoef(y_test, y_pred))
            evlP[k][5] = self.multiclass_roc_auc_score(y_test, y_pred)

            index_counter = 0
            for index in test_index:
                if (y_pred[index_counter]!=y[index]):
                    misindex.append(index)
                index_counter+=1

            k += 1

        average = np.matrix(evlP)
        average = average.mean(axis=0)
        return np.array(average[0]), misindex

    def __GroupCrossFoldKarim(self,X, Y, group, nfold):
        df = pd.DataFrame(X)
        df['groups'] = group
        df['Y'] = Y
        groups = df.groupby(['groups'])
        trainX = []
        testX = []
        trainY = []
        testY = []

        # make array of empty for each fold
        for i in range(nfold):
            trainX.append([])
            testX.append([])
            trainY.append([])
            testY.append([])
        # cross fold for each of groups and collect all of them
        for group in groups:
            data = np.array(group)[1].to_numpy()
            print ("Data size {}".format(data.shape))
            kf = StratifiedKFold(n_splits=nfold)
            if (data.shape[0]<=5):
                kf = StratifiedKFold(n_splits=2)

            datax = data[:, 0:data.shape[1] - 2]
            datay = data[:, data.shape[1] - 1].astype(np.int32)

            index = 0
            print(type_of_target(datay))
            for train_index, test_index in kf.split(X=datax,y= datay):
                #       print ("{},{}".format(datax[train_index,:].shape[0],datax[test_index,:].shape[0]))

                for itemrow, ylabel in zip(datax[train_index, :], datay[train_index]):
                    # print (itemrow)
                    trainX[index].append(itemrow)
                    trainY[index].append(ylabel)
                for itemrow, ylabel in zip(datax[test_index, :], datay[test_index]):
                    # print (itemrow)
                    testX[index].append(itemrow)
                    testY[index].append(ylabel)

                index += 1

        # convert each fold array to numpy box array
        for i in range(nfold):
            trainX[i] = np.array(trainX[i])
            testX[i] = np.array(testX[i])
            trainY[i] = np.array(trainY[i])
            testY[i] = np.array(testY[i])
        return trainX, trainY, testX, testY

    def __Cross_Validation_Group(self, X, y,groups, classifier, kf):
        evlP = [[0 for x in range(6)] for YY in range(self.kfold)]
        k = 0
        data1, data2, data3, data4 = self.__GroupCrossFoldKarim(X=X,Y=y,group=groups,nfold= self.kfold)
        for trainX, trainY, testX, testY in zip(data1, data2, data3, data4):
            classifier.fit(trainX, trainY)
            y_pred = classifier.predict(testX)
            y_test = testY

            evlP[k][0] = (precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred)))
            evlP[k][1] = (f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred)))
            evlP[k][2] = (accuracy_score(y_test, y_pred))
            evlP[k][3] = (recall_score(y_test, y_pred, average="weighted"))
            evlP[k][4] = (matthews_corrcoef(y_test, y_pred))
            evlP[k][5] = self.multiclass_roc_auc_score(y_test, y_pred)
            k += 1

        average = np.matrix(evlP)
        average = average.mean(axis=0)
        return np.array(average[0])
    def __TrainAndTest(self, xTrain, yTrain, xTest, yTest, classifier):
        evlP = np.zeros(6)
        classifier.fit(xTrain, yTrain)
        y_pred = classifier.predict(xTest)
        y_test = yTest

        evlP[0] = (precision_score(y_test, y_pred, average='micro'))
        evlP[1] = (f1_score(y_test, y_pred, average='macro'))
        evlP[2] = (accuracy_score(y_test, y_pred))
        evlP[3] = (recall_score(y_test, y_pred, average="weighted"))
        evlP[4] = (matthews_corrcoef(y_test, y_pred))
        evlP[5] = self.multiclass_roc_auc_score(y_test, y_pred)
        return evlP

    def DoCrossValidation(self, X,Y,resultfilename,savetofile=True):
        run_times = 1
        overall_performance = []
        misindex = []
        for runs in range(run_times):
            kf = StratifiedKFold(n_splits=self.kfold)
            classifier_names = []
            total_performance = []
            for classifier in self.classifier:
                classifier_names.append(classifier.__class__.__name__)
                performance,misindex = self.__Cross_Validation(X,Y,classifier,kf)
                total_performance.append(performance[0])

            total_performance = np.array(total_performance)
            overall_performance.append(total_performance)

        overall_performance = np.array(overall_performance)
        overal_mean = np.mean(overall_performance,axis=0)
        overal_variance = np.std(overall_performance,axis=0)
        if (savetofile==True):
            df1 = pd.DataFrame(overal_mean,index = classifier_names,columns=self.evaluationName)
            df2 = pd.DataFrame(overal_variance,index = classifier_names,columns=self.evaluationName)
            with pd.ExcelWriter('result/Cross_{}.xls'.format(resultfilename)) as writer:
                df1.to_excel(writer, sheet_name='Mean')
                df2.to_excel(writer, sheet_name='Variance')
        else:
            return overal_mean,overal_variance,misindex

    def DoCrossValidationGroup(self, X,Y,groups,resultfilename,savetofile=True):
        run_times = 1
        overall_performance = []
        for runs in range(run_times):
            kf = StratifiedKFold(n_splits=self.kfold)
            classifier_names = []
            total_performance = []
            for classifier in self.classifier:
                classifier_names.append(classifier.__class__.__name__)
                performance = self.__Cross_Validation_Group(X,Y,groups=groups,classifier=classifier,kf=kf)
                total_performance.append(performance[0])
            total_performance = np.array(total_performance)
            overall_performance.append(total_performance)

        overall_performance = np.array(overall_performance)
        overal_mean = np.mean(overall_performance,axis=0)
        overal_variance = np.std(overall_performance,axis=0)
        if (savetofile==True):
            df1 = pd.DataFrame(overal_mean,index = classifier_names,columns=self.evaluationName)
            df2 = pd.DataFrame(overal_variance,index = classifier_names,columns=self.evaluationName)
            with pd.ExcelWriter('result/Cross_{}.xls'.format(resultfilename)) as writer:
                df1.to_excel(writer, sheet_name='Mean')
                df2.to_excel(writer, sheet_name='Variance')
        else:
            return overal_mean,overal_variance

    def DoLeaveOneOut(self, X,Y,resultfilename,savetofile=True):
        run_times = 10
        overall_performance = []
        for runs in range(run_times):
            indices = np.arange(X.shape[0])

            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, Y, indices, test_size=0.2,random_state=random.randint(1,100))
            X_train = X[idx_train,:]
            y_train = Y[idx_train]
            X_test= X[idx_test,:]
            y_test= Y[idx_test]

            classifier_names = []
            total_performance = []
            for classifier in self.classifier:
                classifier_names.append(classifier.__class__.__name__)
                performance = self.__TrainAndTest(X_train,y_train,X_test,y_test,classifier)
                total_performance.append(performance)
            total_performance = np.array(total_performance)
            overall_performance.append(total_performance)
        overall_performance = np.array(overall_performance)
        overal_mean = np.mean(overall_performance,axis=0)
        overal_variance = np.std(overall_performance,axis=0)
        if (savetofile==True):
            df1 = pd.DataFrame(overal_mean,index = classifier_names,columns=self.evaluationName)
            df2 = pd.DataFrame(overal_variance,index = classifier_names,columns=self.evaluationName)
            with pd.ExcelWriter('result/Leave_One_{}.xls'.format(resultfilename)) as writer:
                df1.to_excel(writer, sheet_name='Mean')
                df2.to_excel(writer, sheet_name='Variance')
        else:
            return overal_mean,overal_variance

    def DoDeepLearning(self,X,Y):
        n_spilit = 5
        skf = StratifiedKFold(n_splits=n_spilit)
        total = []
        evlP = [[0 for x in range(6)] for YY in range(self.kfold)]
        k=0

        for train_index, test_index in skf.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            #y_test = tf.keras.utils.to_categorical(y_test)
            #y_train = tf.keras.utils.to_categorical(y_train)

            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(6000, input_shape=(X_train.shape[1],),
                                            activation='relu',
                                            kernel_initializer='glorot_normal',
                                            kernel_regularizer=tf.keras.regularizers.l1(0.0001))
                      )

            model.add(tf.keras.layers.Dropout(0.4))

            model.add(tf.keras.layers.Dense(500, activation='relu',
                                            use_bias=True,
                                            kernel_initializer='glorot_normal',
                                            kernel_regularizer=tf.keras.regularizers.l1(0.0001))
                      )

            model.add(tf.keras.layers.Dropout(0.2))

            model.add(tf.keras.layers.Dense(100, activation='relu',
                                            use_bias=True,
                                            kernel_initializer='glorot_normal',
                                            kernel_regularizer=tf.keras.regularizers.l1(0.0001))
                      )

            model.add(tf.keras.layers.Dropout(0.4))

            model.add(tf.keras.layers.Dense(70, activation='relu',
                                            use_bias=True,
                                            kernel_initializer='glorot_normal',
                                            kernel_regularizer=tf.keras.regularizers.l1(0.0001))
                      )

            model.add(tf.keras.layers.Dropout(0.4))

            model.add(tf.keras.layers.Dense(30, activation='relu',
                                            use_bias=True,
                                            kernel_initializer='glorot_normal',
                                            kernel_regularizer=tf.keras.regularizers.l1(0.0001))
                      )
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

            # model.compile(optimizer='Adam', loss='bce')
            optimizer = Adam(lr=0.001)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.AUC()])
            # model.summary()
            # This builds the model for the first time:

            model.fit(X_train, y_train, batch_size=705, epochs=25, validation_data=(X_test, y_test))

            y_pred = np.round(model.predict(X_test),0)
            y_temp = []
            for item in y_pred:
                if (item[0]==1):
                    y_temp.append(1)
                else:
                    y_temp.append(0)
            y_pred = np.array(y_temp)

            #y_test = np.argmax(y_test,axis=1)

            evlP[k][0] = (precision_score(y_test, y_pred, average='macro',labels=np.unique(y_pred)))
            evlP[k][1] = (f1_score(y_test, y_pred, average='macro',labels=np.unique(y_pred)))
            evlP[k][2] = (accuracy_score(y_test, y_pred))
            evlP[k][3] = (recall_score(y_test, y_pred, average="weighted"))
            evlP[k][4] = (matthews_corrcoef(y_test, y_pred))
            evlP[k][5] = self.multiclass_roc_auc_score(y_test, y_pred)

            k+=1

        average = np.matrix(evlP)
        average1 = average.mean(axis=0)
        var = average.var(axis=0)
        return average1,var
    def DoDeepLearning1(self,X,Y):
        n_spilit = 5
        skf = StratifiedKFold(n_splits=n_spilit)
        total = []
        evlP = [[0 for x in range(6)] for YY in range(self.kfold)]
        k=0
        #X3 = np.reshape(X3, (len(X3), feature_count - 2, 1))
        X = np.reshape(X, (len(X), X.shape[1], 1))
        for train_index, test_index in skf.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            #y_test = tf.keras.utils.to_categorical(y_test)
            #y_train = tf.keras.utils.to_categorical(y_train)

            myinput = tf.keras.layers.Input(shape=(X_train.shape[1],1),)

            model = tf.keras.layers.Conv1D(filters=1000 , kernel_size= 4 , activation='relu', padding='same',)(myinput)
            model = tf.keras.layers.MaxPool1D(pool_size=2)(model)
            model = tf.keras.layers.Dropout(0.2)(model)

            model = tf.keras.layers.Conv1D(filters=150 , kernel_size= 5 , activation='relu', padding='same')(model)
            model = tf.keras.layers.MaxPool1D(pool_size=2)(model)
            model = tf.keras.layers.Dropout(0.2)(model)

            model = tf.keras.layers.Conv1D(filters=100 , kernel_size= 5 , activation='relu', padding='same')(model)
            model = tf.keras.layers.MaxPool1D(pool_size=2)(model)
            model = tf.keras.layers.Dropout(0.2)(model)

            model = tf.keras.layers.Conv1D(filters=50 , kernel_size= 5 , activation='relu', padding='same')(model)
            model = tf.keras.layers.MaxPool1D(pool_size=2)(model)
            model = tf.keras.layers.Dropout(0.2)(model)
            model = tf.keras.layers.Dense(30, activation='relu')(model)

            flatten = tf.keras.layers.Flatten()(model)
            output = tf.keras.layers.Dense(1, activation='sigmoid')(flatten)

            model = tf.keras.Model(myinput,output)
            model.summary()
            #sigmoid change to softmax withh Dense 2
            # model.compile(optimizer='Adam', loss='bce')
            optimizer = Adam(lr=0.001)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'],beta1=0.9,beta2=0.99) #categorical_crossentropy change to binary_crossentropy
            model.fit(X_train, y_train, batch_size=704, epochs=150,validation_data=(X_test,y_test))

            y_pred = np.round(model.predict(X_test),0)
            y_temp = []
            for item in y_pred:
                if (item[0]==1):
                    y_temp.append(1)
                else:
                    y_temp.append(0)
            y_pred = np.array(y_temp)

            #y_test = np.argmax(y_test,axis=1)

            evlP[k][0] = (precision_score(y_test, y_pred, average='macro',labels=np.unique(y_pred)))
            evlP[k][1] = (f1_score(y_test, y_pred, average='macro',labels=np.unique(y_pred)))
            evlP[k][2] = (accuracy_score(y_test, y_pred))
            evlP[k][3] = (recall_score(y_test, y_pred, average="weighted"))
            #evlP[k][4] = (recall_score(y_test, y_pred, average="weighted"))
            #evlP[k][5] = (recall_score(y_test, y_pred, average="weighted"))
            evlP[k][4] = (matthews_corrcoef(y_test, y_pred))
            evlP[k][5] = self.multiclass_roc_auc_score(y_test, y_pred)
            k+=1

        average = np.matrix(evlP)
        average1 = average.mean(axis=0)
        var = average.var(axis=0)
        return average1,var

    def GridSearchSVC(self,X,Y):
        from sklearn.model_selection import GridSearchCV
        param_grid = {'C': [0.1, 1,2,5, 10,30,50, 100], 'gamma': [1, 0.7, 0.5, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        indices = np.arange(X.shape[0])
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, Y, indices, test_size=0.2, random_state=random.randint(1, 100))
        grid.fit(X_train, y_train)
        print(grid.best_params_)
        print("****************************")
        print(grid.best_estimator_)
    def GridSearchRidgeClassifier(self, X, Y):
        from sklearn.model_selection import GridSearchCV
        params = []
        for i in range(100):
            params.append(random.random())

        param_grid = {'alpha': params,'solver':['svd','cholesky','lsqr','sag','lbfgs']}
        grid = GridSearchCV(RidgeClassifier(), param_grid, refit=True, verbose=2)
        indices = np.arange(X.shape[0])
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, Y, indices, test_size=0.2,
                                                                                 random_state=random.randint(1,
                                                                                                             100))
        grid.fit(X_train, y_train)
        print(grid.best_params_)
        print("****************************")
        print(grid.best_estimator_)
    def GridSearchKNNClassifier(self, X, Y):
        from sklearn.model_selection import GridSearchCV
        params = []
        for i in range(100):
            params.append(random.random())

        param_grid = {'n_neighbors': [3,4,5,6,7,8,9,10,15,20,30],
                      'leaf_size':[1,5,10,20,30,40,50,100],
                      'metric':['euclidean','manhattan']}
        grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=2)
        indices = np.arange(X.shape[0])
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, Y, indices, test_size=0.2,
                                                                                 random_state=random.randint(1,
                                                                                                             100))
        grid.fit(X_train, y_train)
        print(grid.best_params_)
        print("****************************")
        print(grid.best_estimator_)
    def GridSearchRandomForest(self, X, Y):
        from sklearn.model_selection import GridSearchCV
        params = []
        for i in range(100):
            params.append(random.random())
        param_grid={'bootstrap': [True, False],
         'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
         'max_features': ['auto', 'sqrt'],
         'min_samples_leaf': [1, 2, 4],
         'min_samples_split': [2, 5, 10],
         'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
        grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=1)

        grid.fit(X,Y)
        print(grid.best_params_)
        print("****************************")
        print(grid.best_estimator_)
#****************************PreProcess*****************************
class Preprocess():
    def __init__(self):
        pass
    def CheckLabel(self,name):
        for item in phenotype:
            if (item[11]==name):
                classlabel = item[5]
                sex = item[6]
                age = item[7]
                siteid = item[10]
                return age,sex,classlabel,siteid
        return None,None,None,None
    def Get_Upper_Triangle(self,data):
        x = []
        for i in range(data.shape[0]):
            for j in range(i-1):
                x.append(data[i,j])
        x= np.array(x)
        return x
    def Drop_Constant_Columns(self,X):
        dataframe = pd.DataFrame(X)
        for column in dataframe.columns:
            if len(dataframe[column].unique()) == 1:
                dataframe.drop(column,inplace=True,axis=1)
        return dataframe
    def Vectorize_Matrix(self,X):
        data = []
        for item1 in X:
            for item2 in item1:
                data.append(item2)
        return np.array(data)
    def Drop_Correlated(self,X):
        df = pd.DataFrame(X)

        corr_matrix  = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

        df.drop(to_drop, axis=1, inplace=True)
        return df.to_numpy()

    def Read_1D(self,filename):

        file = open(filename, "r")
        file.readline()
        lines = file.readlines()

        roi_signal = []
        for item in lines:
            row = item.split("\t")

            row = np.asarray(row)

            row = row.astype(float)

            roi_signal.append(row)
        roi_signal = np.array(roi_signal)

        file.close()
        return roi_signal
    def Load_Atlas_Data_Abide(self,dirpath,fromage=1,toage=60,sitename='all',norm="roboust",applythresould =False,dropcorrelated = False,dropconstant=False):
        X=[]
        Y=[]
        Age = []
        Sex = []
        Site = []
        for file in glob.glob(dirpath):
            basename = os.path.basename(file)
            _index = basename.find("_rois")
            filesitename= basename[0:_index]
            #print(filesitename)
            age,sex,classlabel,siteid = self.CheckLabel(filesitename)
            if (age!=None):
                if (sitename=='all'):
                    if ( age>=fromage and age<=toage):
                        time_series = self.Read_1D(file)
                        correlation_measure = ConnectivityMeasure(kind='correlation')
                        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
                        correlation_matrix = self.Get_Upper_Triangle(correlation_matrix)
                        X.append(correlation_matrix)

                        Age.append(age)
                        Sex.append(sex)
                        Site.append(siteid)

                        if (classlabel=="Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
                else:
                    if ( age >= fromage and age<=toage and siteid==sitename):
                        time_series = self.Read_1D(file)
                        correlation_measure = ConnectivityMeasure(kind='correlation')
                        correlation_matrix = correlation_measure.fit_transform([time_series])[0]
                        correlation_matrix = self.Get_Upper_Triangle(correlation_matrix)
                        X.append(correlation_matrix)

                        Age.append(age)
                        Sex.append(sex)
                        Site.append(siteid)
                        if (classlabel=="Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
        X = np.array(X)
        Age = np.array(Age)
        Sex = np.array(Sex)
        Site = np.array(Site)

        # Do some useful preprocess such as constnat removal, correlated features removal and low variance feature removal

        print ("Main Data Size : {} . {}".format(X.shape[0],X.shape[1]))

        if (dropconstant==True):
            X = self.Drop_Constant_Columns(X)
            print("After Remove Contant : {} . {}".format(X.shape[0], X.shape[1]))

        if (dropcorrelated==True):
            X = self.Drop_Correlated(X)
            print("After Remove Correlated : {} . {}".format(X.shape[0], X.shape[1]))

        if (applythresould==True):
            variancemodel = VarianceThreshold(threshold=(0.04))
            X = variancemodel.fit_transform(X)
            print("After Apply Variance Thresould : {} . {}".format(X.shape[0], X.shape[1]))

        Y = np.array(Y)

        if (norm == "stand"):
            X = StandardScaler().fit_transform(X)
        elif(norm == "roboust"):
            X = RobustScaler().fit_transform(X)
        elif(norm == "minmax"):
            X = RobustScaler().fit_transform(X)

        # make batch_data for neuroCombat and return X,Y, batch .  batch_data is used only for batch_data

        batch_data = pd.DataFrame({"batch": (Site), "gender": (Sex), "age": (Age)})
        return X, Y, batch_data
    def Load_Atlas_Data_NiLearn(self,dirpath,fromage=1,toage=60,sitename='all',norm="roboust",applythresould =False,dropcorrelated = False,dropconstant=False):
        X=[]
        Y=[]
        Age = []
        Sex = []
        Site = []
        for file in glob.glob(dirpath):
            basename = os.path.basename(file)
            _index = basename.find("_func_pre")
            filesitename= basename[0:_index]
            #print(filesitename)
            age,sex,classlabel,siteid = self.CheckLabel(filesitename)
            if (age!=None):
                if (sitename=='all'):
                    if ( age>=fromage and age<=toage):
                        correlation_matrix =np.load(file)
                        correlation_matrix = self.Get_Upper_Triangle(correlation_matrix)
                        X.append(correlation_matrix)

                        Age.append(age)
                        Sex.append(sex)
                        Site.append(siteid)

                        if (classlabel=="Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
                else:
                    if ( age >= fromage and age<=toage and siteid==sitename):
                        correlation_matrix =np.load(file)
                        correlation_matrix = self.Get_Upper_Triangle(correlation_matrix)
                        X.append(correlation_matrix)

                        Age.append(age)
                        Sex.append(sex)
                        Site.append(siteid)
                        if (classlabel=="Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
        X = np.array(X)
        Age = np.array(Age)
        Sex = np.array(Sex)
        Site = np.array(Site)

        # Do some useful preprocess such as constnat removal, correlated features removal and low variance feature removal

        print ("Main Data Size : {} . {}".format(X.shape[0],X.shape[1]))

        if (dropconstant==True):
            X = self.Drop_Constant_Columns(X)
            print("After Remove Contant : {} . {}".format(X.shape[0], X.shape[1]))

        if (dropcorrelated==True):
            X = self.Drop_Correlated(X)
            print("After Remove Correlated : {} . {}".format(X.shape[0], X.shape[1]))

        if (applythresould==True):
            variancemodel = VarianceThreshold(threshold=(0.04))
            X = variancemodel.fit_transform(X)
            print("After Apply Variance Thresould : {} . {}".format(X.shape[0], X.shape[1]))

        Y = np.array(Y)

        if (norm == "stand"):
            X = StandardScaler().fit_transform(X)
        elif(norm == "roboust"):
            X = RobustScaler().fit_transform(X)
        elif(norm == "minmax"):
            X = RobustScaler().fit_transform(X)

        # make batch_data for neuroCombat and return X,Y, batch .  batch_data is used only for batch_data

        batch_data = pd.DataFrame({"batch": (Site), "gender": (Sex), "age": (Age)})
        return X, Y, batch_data

    def Load_Data(self,dirpath,fromage=1,toage=60,sitename='all',norm="roboust",applythresould =False,dropcorrelated = False,dropconstant=False):

        X=[]
        Y=[]
        Age = []
        Sex = []
        Site = []

        for file in glob.glob(dirpath):
            basename = os.path.basename(file)
            filesitename= basename[0:len(basename)-17]
            print(filesitename)
            age,sex,classlabel,siteid = self.CheckLabel(filesitename)
            if (age!=None):
                if (sitename=='all'):
                    if ( age>=fromage and age<=toage):
                        X1 = np.load(file)
                        X1 = self.Get_Upper_Triangle(X1)
                        X.append(X1)
                        Age.append(age)
                        Sex.append(sex)
                        Site.append(siteid)
                        if (classlabel=="Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
                else:
                    if ( age >= fromage and age<=toage and siteid==sitename):
                        X1 = np.load(file)
                        X1 = self.Get_Upper_Triangle(X1)
                        X.append(X1)
                        Age.append(age)
                        Sex.append(sex)
                        Site.append(siteid)
                        if (classlabel=="Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
        X = np.array(X)
        Age = np.array(Age)
        Sex = np.array(Sex)
        Site = np.array(Site)

        # Do some useful preprocess such as constnat removal, correlated features removal and low variance feature removal

        print ("Main Data Size : {} . {}".format(X.shape[0],X.shape[1]))

        if (dropconstant==True):
            X = self.Drop_Constant_Columns(X)
            print("After Remove Contant : {} . {}".format(X.shape[0], X.shape[1]))

        if (dropcorrelated==True):
            X = self.Drop_Correlated(X)
            print("After Remove Correlated : {} . {}".format(X.shape[0], X.shape[1]))

        if (applythresould==True):
            variancemodel = VarianceThreshold(threshold=(.8 * (1 - .8)))
            X = variancemodel.fit_transform(X)
            print("After Apply Variance Thresould : {} . {}".format(X.shape[0], X.shape[1]))

        Y = np.array(Y)

        if (norm == "stand"):
            X = StandardScaler().fit_transform(X)
        elif(norm == "roboust"):
            X = RobustScaler().fit_transform(X)
        elif(norm == "minmax"):
            X = RobustScaler().fit_transform(X)

        # make batch_data for neuroCombat and return X,Y, batch .  batch_data is used only for batch_data

        batch_data = pd.DataFrame({"batch": (Site), "gender": (Sex), "age": (Age), "disease": (Y)})
        return X, Y, batch_data
    def LoadMatData(self,dirpath):
        data_dict = mat73.loadmat(dirpath)
        X1 = np.asarray(data_dict['windows']['X'])[3:]
        Y1 = np.asarray(data_dict['windows']['Y'])[3:]

        X = []
        Y = []
        i = 0
        for item1,item2 in zip(X1,Y1):
            if (i==0 or i==1 or i==2):
                i+=1
                continue
            row= item1
            #label = item2[0]
            X.append(row)
            if (item2==1):
                Y.append(1)
            else:
                Y.append(0)
            i+=1
        X = np.array(X)
        Y = np.array(Y)

        del X1,Y1,item1,item2,row

        X = X.astype(float)
        X = MinMaxScaler().fit_transform(X)
        Y = Y.astype(float)
        return X,Y
    def Load_data_wavelet(self,dirpath,fromage=1,toage=60,sitename='UM_1',norm="roboust"):
        X=[]
        Y=[]
        for file in glob.glob(dirpath):
            basename = os.path.basename(file)
            base1= basename[0:len(basename)-17]
            print(base1)
            age,sex,classlabel,siteid = self.CheckLabel(base1)
            if (age!=None):
                if (sitename=='all'):
                    if ( age>=fromage and age<=toage ):
                        X1 = np.load(file)
                        #X1 = Get_Upper_Triangle(X1)
                        X.append(X1)
                        if (classlabel=="Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
                else:
                    if ( age >= fromage and age<=toage and siteid==sitename):
                        X1 = np.load(file)
                        #X1 = Get_Upper_Triangle(X1)
                        X.append(X1)
                        if (classlabel=="Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
        X = np.array(X)
        print ("Main Data Size : {} . {}".format(X.shape[0],X.shape[1]))
    #    X = Drop_Constant_Columns(X)
        print("After Remove Contant : {} . {}".format(X.shape[0], X.shape[1]))
        #X = Drop_Correlated(X)
        print("After Remove Correlated : {} . {}".format(X.shape[0], X.shape[1]))
        Y = np.array(Y)

        if (norm == "stand"):
            X = StandardScaler().fit_transform(X)
        elif(norm == "roboust"):
            X = RobustScaler().fit_transform(X)
        elif(norm == "minmax"):
            X = RobustScaler().fit_transform(X)
        return X,Y
    def BinAge(self,age):
        if (age >= 5 and age<=10):
            return "5-10"
        elif(age>10 and age<=20):
            return "10-20"
        elif(age>20 and age<=40):
            return "20-40"
        else:
            return ">40"

    def Load_data_wavelet_2d(self,dirpath,fromage=1,toage=60,sitename='all',norm="minmax",applythresould =False,dropcorrelated = False,dropconstant=False):
        X=[]
        Y=[]
        Age = []
        Sex = []
        Site = []
        for file in glob.glob(dirpath):
            basename = os.path.basename(file)
            _index = basename.find("_func_")
            filesitename = basename[0:_index]
            age,sex,classlabel,siteid = self.CheckLabel(filesitename)
            if (age!=None):
                if (sitename=='all'):
                    if ( age>=fromage and age<=toage ):
                        X1 = np.load(file)
                        X1 = self.Vectorize_Matrix(X1)
                        X.append(X1)
                        Age.append(self.BinAge(age))
                        Sex.append(sex)
                        Site.append(siteid)
                        if (classlabel=="Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
                else:
                    if ( age >= fromage and age<=toage and siteid==sitename):
                        X1 = np.load(file)
                        X1 = self.Vectorize_Matrix(X1)
                        X.append(X1)
                        Age.append(self.BinAge(age))
                        Sex.append(sex)
                        Site.append(siteid)
                        if (classlabel=="Autism"):
                            Y.append(1)
                        else:
                            Y.append(0)
        X = np.array(X)

        Age = np.array(Age)
        Sex = np.array(Sex)
        Site = np.array(Site)

        print ("Main Data Size : {} . {}".format(X.shape[0],X.shape[1]))

        if (dropconstant==True):
            X = self.Drop_Constant_Columns(X)
            print("After Remove Contant : {} . {}".format(X.shape[0], X.shape[1]))

        if (dropcorrelated==True):
            X = self.Drop_Correlated(X)
            print("After Remove Correlated : {} . {}".format(X.shape[0], X.shape[1]))

        if (applythresould==True):
            variancemodel = VarianceThreshold(threshold=(0.04))
            X = variancemodel.fit_transform(X)
            print("After Apply Variance Thresould : {} . {}".format(X.shape[0], X.shape[1]))


        index = np.isnan(X).any(axis=0)
        X = X[:,~index]

        Y = np.array(Y)

        if (norm == "stand"):
            X = StandardScaler().fit_transform(X)
        elif(norm == "roboust"):
            X = RobustScaler().fit_transform(X)
        elif(norm == "minmax"):
            X = RobustScaler().fit_transform(X)

        batch_data = pd.DataFrame({"batch": (Site), "gender": (Sex), "age": (Age)})
        return X, Y, batch_data

#****************************Make Models*****************************

class AutismClassifiers():
    def Classify_Atlas(self,resultfilename,fromage,toage):
        #This function classify abide autism fmri dataset with 13 classifier methods and save the perfomance of each classifier
        # in the excel file located in the result folder, so before running this code, it is need to create a folder with result name

        preproc =Preprocess()

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifier_names = classifier.Get_Classifier_Names()

        atlas_names = []
        data_dir = "Atlas_Extractor/Abide/*"
        for index,folder in enumerate(glob.glob(data_dir)):
            atlas_names.append(folder.split("\\")[1])

        data_dir_nilearn = "Atlas_Extractor/Nilearn/*"
        for index,folder in enumerate(glob.glob(data_dir_nilearn)):
            atlas_names.append(folder.split("\\")[1])

        accuracy =np.zeros((len(classifier_names),len(atlas_names)))
        precision =np.zeros((len(classifier_names),len(atlas_names)))
        recall =np.zeros((len(classifier_names),len(atlas_names)))
        auc =np.zeros((len(classifier_names),len(atlas_names)))

        #Train Abide Atlas
        lastindex = 0
        for index,folder in enumerate(glob.glob(data_dir)):
            print (folder)
            files = folder + "/*.1d"
            X, Y, df = preproc.Load_Atlas_Data_Abide(files, fromage, toage,applythresould=False)
            mean, varivance = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

            accuracy[:,index] = mean[:,2]
            precision[:,index] = mean[:,0]
            recall[:,index] = mean[:,3]
            auc[:,index] = mean[:,5]
            lastindex=index

        # Train Nilearn Atlas and save result in the same numpy array accuracy,precision, auc, recall
        for index, folder in enumerate(glob.glob(data_dir_nilearn)):
            print(folder)
            files = folder + "/*.npy"
            X, Y, df = preproc.Load_Atlas_Data_NiLearn(files, fromage, toage,applythresould=False)
            mean, varivance = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

            accuracy[:, index+lastindex] = mean[:, 2]
            precision[:, index+lastindex] = mean[:, 0]
            recall[:, index+lastindex] = mean[:, 3]
            auc[:, index+lastindex] = mean[:, 5]

        df1 = pd.DataFrame(accuracy, index=classifier_names, columns=atlas_names)
        df2 = pd.DataFrame(precision, index=classifier_names, columns=atlas_names)
        df3 = pd.DataFrame(recall, index=classifier_names, columns=atlas_names)
        df4 = pd.DataFrame(auc, index=classifier_names, columns=atlas_names)

        with pd.ExcelWriter('result/Cross_{}.xls'.format(resultfilename)) as writer:
            df1.to_excel(writer, sheet_name='Accuracy')
            df2.to_excel(writer, sheet_name='Precision')
            df3.to_excel(writer, sheet_name='Recall')
            df4.to_excel(writer, sheet_name='Auc')
    def Classify_Atlas_With_Combat(self,resultfilename,fromage,toage):
        #This function classify abide autism fmri dataset with 13 classifier methods and save the perfomance of each classifier
        # in the excel file located in the result folder, so before running this code, it is need to create a folder with result name

        preproc =Preprocess()

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifier_names = classifier.Get_Classifier_Names()

        atlas_names = []
        data_dir = "Atlas_Extractor/Abide/*"
        for index,folder in enumerate(glob.glob(data_dir)):
            atlas_names.append(folder.split("\\")[1])

        data_dir_nilearn = "Atlas_Extractor/Nilearn/*"
        for index,folder in enumerate(glob.glob(data_dir_nilearn)):
            atlas_names.append(folder.split("\\")[1])

        accuracy =np.zeros((len(classifier_names),len(atlas_names)))
        precision =np.zeros((len(classifier_names),len(atlas_names)))
        recall =np.zeros((len(classifier_names),len(atlas_names)))
        auc =np.zeros((len(classifier_names),len(atlas_names)))

        #Train Abide Atlas
        lastindex = 0

        for index,folder in enumerate(glob.glob(data_dir)):
            print (folder)
            files = folder + "/*.1d"

            X, Y, covars = preproc.Load_Atlas_Data_Abide(files, fromage,toage,applythresould=False)
            categorical_cols = ['gender', 'age']
            batch_col = 'batch'
            X = np.transpose(X)
            data_combat = neuroCombat(dat=X, covars=covars, batch_col=batch_col, categorical_cols=categorical_cols)["data"]
            #data_combat = neuroCombat(dat=X, covars=covars, categorical_cols=categorical_cols)["data"]
            X = np.transpose(data_combat)
            mean, varivance = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

            accuracy[:,index] = mean[:,2]
            precision[:,index] = mean[:,0]
            recall[:,index] = mean[:,3]
            auc[:,index] = mean[:,5]
            lastindex=index

        # Train Nilearn Atlas and save result in the same numpy array accuracy,precision, auc, recall
        #for index, folder in enumerate(glob.glob(data_dir_nilearn)):
        #    print(folder)
        #    files = folder + "/*.npy"
        #    X, Y, covars = preproc.Load_Atlas_Data_NiLearn(files, fromage,toage,applythresould=False)
        #    X = np.transpose(X)
        #    categorical_cols = ['gender', 'age']
        #    batch_col = 'batch'
        #    data_combat = neuroCombat(dat=X, covars=covars, batch_col=batch_col, categorical_cols=categorical_cols)["data"]
            #data_combat = neuroCombat(dat=X, covars=covars, categorical_cols=categorical_cols)["data"]
        #    X = np.transpose(data_combat)
        #    mean, varivance = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

            #accuracy[:, index+lastindex] = mean[:, 2]
            #precision[:, index+lastindex] = mean[:, 0]
            #recall[:, index+lastindex] = mean[:, 3]
            #auc[:, index+lastindex] = mean[:, 5]

        df1 = pd.DataFrame(accuracy, index=classifier_names, columns=atlas_names)
        df2 = pd.DataFrame(precision, index=classifier_names, columns=atlas_names)
        df3 = pd.DataFrame(recall, index=classifier_names, columns=atlas_names)
        df4 = pd.DataFrame(auc, index=classifier_names, columns=atlas_names)

        with pd.ExcelWriter('result/Cross_{}.xls'.format(resultfilename)) as writer:
            df1.to_excel(writer, sheet_name='Accuracy')
            df2.to_excel(writer, sheet_name='Precision')
            df3.to_excel(writer, sheet_name='Recall')
            df4.to_excel(writer, sheet_name='Auc')
    def Classify_Atlas_GroupKFold(self,resultfilename,fromage,toage):
        #This function classify abide autism fmri dataset with 13 classifier methods and save the perfomance of each classifier
        # in the excel file located in the result folder, so before running this code, it is need to create a folder with result name

        preproc =Preprocess()

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifier_names = classifier.Get_Classifier_Names()

        atlas_names = []
        data_dir = "Atlas_Extractor/Abide/*"
        for index,folder in enumerate(glob.glob(data_dir)):
            atlas_names.append(folder.split("\\")[1])

        data_dir_nilearn = "Atlas_Extractor/Nilearn/*"
        for index,folder in enumerate(glob.glob(data_dir_nilearn)):
            atlas_names.append(folder.split("\\")[1])

        accuracy =np.zeros((len(classifier_names),len(atlas_names)))
        precision =np.zeros((len(classifier_names),len(atlas_names)))
        recall =np.zeros((len(classifier_names),len(atlas_names)))
        auc =np.zeros((len(classifier_names),len(atlas_names)))

        #Train Abide Atlas
        lastindex = 0

        for index,folder in enumerate(glob.glob(data_dir)):
            print (folder)
            files = folder + "/*.1d"

            X, Y, covars = preproc.Load_Atlas_Data_Abide(files, fromage,toage,applythresould=False)
            groups = covars.to_numpy()
            group = groups[:,0]
            mean, varivance = classifier.DoCrossValidationGroup(X, Y,group, "filename", savetofile=False)

            accuracy[:,index] = mean[:,2]
            precision[:,index] = mean[:,0]
            recall[:,index] = mean[:,3]
            auc[:,index] = mean[:,5]
            lastindex=index

        # Train Nilearn Atlas and save result in the same numpy array accuracy,precision, auc, recall
        for index, folder in enumerate(glob.glob(data_dir_nilearn)):
            print(folder)
            files = folder + "/*.npy"
            X, Y, covars = preproc.Load_Atlas_Data_NiLearn(files, fromage,toage,applythresould=False)
            groups = covars.to_numpy()
            group = groups[:,0]
            mean, varivance = classifier.DoCrossValidationGroup(X, Y, group,"filename", savetofile=False)

            accuracy[:, index+lastindex] = mean[:, 2]
            precision[:, index+lastindex] = mean[:, 0]
            recall[:, index+lastindex] = mean[:, 3]
            auc[:, index+lastindex] = mean[:, 5]

        df1 = pd.DataFrame(accuracy, index=classifier_names, columns=atlas_names)
        df2 = pd.DataFrame(precision, index=classifier_names, columns=atlas_names)
        df3 = pd.DataFrame(recall, index=classifier_names, columns=atlas_names)
        df4 = pd.DataFrame(auc, index=classifier_names, columns=atlas_names)

        with pd.ExcelWriter('result/Cross_GroupFold{}.xls'.format(resultfilename)) as writer:
            df1.to_excel(writer, sheet_name='Accuracy')
            df2.to_excel(writer, sheet_name='Precision')
            df3.to_excel(writer, sheet_name='Recall')
            df4.to_excel(writer, sheet_name='Auc')
    def Classify_Atlas_Abide_For_Each_Site(self):
        preproc =Preprocess()
        site_names = np.unique(phenotype[:,10])
        #X, Y, df = preproc.Load_Atlas_Data_Abide(files, 1, 70, applythresould=True)

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifier_names = classifier.Get_Classifier_Names()



        data_dir = "Atlas_Extractor/Abide/*"
        for index,folder in enumerate(glob.glob(data_dir)):
            accuracy = np.zeros((len(classifier_names), len(site_names)))
            precision = np.zeros((len(classifier_names), len(site_names)))
            recall = np.zeros((len(classifier_names), len(site_names)))
            auc = np.zeros((len(classifier_names), len(site_names)))

            for index,site_name in enumerate(site_names):
                files = folder + "/*.1d"
                try:
                    X, Y, covars = preproc.Load_Atlas_Data_Abide(files, 1, 70,sitename=site_name, applythresould=False)
                    mean, varivance = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

                    accuracy[:, index] = mean[:, 2]
                    precision[:, index] = mean[:, 0]
                    recall[:, index] = mean[:, 3]
                    auc[:, index] = mean[:, 5]
                except:
                    pass

                lastindex = index


            df1 = pd.DataFrame(accuracy, index=classifier_names, columns=site_names)
            df2 = pd.DataFrame(precision, index=classifier_names, columns=site_names)
            df3 = pd.DataFrame(recall, index=classifier_names, columns=site_names)
            df4 = pd.DataFrame(auc, index=classifier_names, columns=site_names)

            filename = os.path.split(folder)[1]
            with pd.ExcelWriter('result/Cross_{}.xls'.format(filename)) as writer:
                df1.to_excel(writer, sheet_name='Accuracy')
                df2.to_excel(writer, sheet_name='Precision')
                df3.to_excel(writer, sheet_name='Recall')
                df4.to_excel(writer, sheet_name='Auc')

    def Classify_Atlas_Nilearn_For_Each_Site(self):
        preproc =Preprocess()
        site_names = np.unique(phenotype[:,10])
        #X, Y, df = preproc.Load_Atlas_Data_Abide(files, 1, 70, applythresould=True)

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifier_names = classifier.Get_Classifier_Names()



        data_dir = "Atlas_Extractor/Nilearn/*"
        for index,folder in enumerate(glob.glob(data_dir)):
            accuracy = np.zeros((len(classifier_names), len(site_names)))
            precision = np.zeros((len(classifier_names), len(site_names)))
            recall = np.zeros((len(classifier_names), len(site_names)))
            auc = np.zeros((len(classifier_names), len(site_names)))

            for index,site_name in enumerate(site_names):
                files = folder + "/*.npy"
                try:
                    X, Y, covars = preproc.Load_Atlas_Data_NiLearn(files, 1, 70,sitename=site_name, applythresould=False)
                    mean, varivance = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

                    accuracy[:, index] = mean[:, 2]
                    precision[:, index] = mean[:, 0]
                    recall[:, index] = mean[:, 3]
                    auc[:, index] = mean[:, 5]
                except:
                    pass

                lastindex = index


            df1 = pd.DataFrame(accuracy, index=classifier_names, columns=site_names)
            df2 = pd.DataFrame(precision, index=classifier_names, columns=site_names)
            df3 = pd.DataFrame(recall, index=classifier_names, columns=site_names)
            df4 = pd.DataFrame(auc, index=classifier_names, columns=site_names)

            filename = os.path.split(folder)[1]
            with pd.ExcelWriter('result/Cross_{}.xls'.format(filename)) as writer:
                df1.to_excel(writer, sheet_name='Accuracy')
                df2.to_excel(writer, sheet_name='Precision')
                df3.to_excel(writer, sheet_name='Recall')
                df4.to_excel(writer, sheet_name='Auc')

    def Classify_Atlas_For_Each_Site(self):
        preproc = Preprocess()
        site_names = np.unique(phenotype[:, 10])
        # X, Y, df = preproc.Load_Atlas_Data_Abide(files, 1, 70, applythresould=True)

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifier_names = classifier.Get_Classifier_Names()

        abide_atlas = ['rois_aal','rois_cc200','rois_cc400','rois_dosenbach160','rois_ez','rois_ho','rois_tt']
        nilearn_atlas = ['croddock','difumo128','difumo256','difumo512','hard','msdl','rsmith70','smith70']

        atlas_all = ['rois_aal','rois_cc200','rois_cc400','rois_dosenbach160','rois_ez','rois_ho','rois_tt','croddock','difumo128','difumo256','difumo512','hard','msdl','rsmith70','smith70']

        accuracy = np.zeros((len(atlas_all), len(site_names)))
        precision = np.zeros((len(atlas_all), len(site_names)))
        recall = np.zeros((len(atlas_all), len(site_names)))
        auc = np.zeros((len(atlas_all), len(site_names)))

        data_dir = "Atlas_Extractor/Abide/"
        lastindex = 0
        for i, filter_name in enumerate(abide_atlas):
            for j, site_name in enumerate(site_names):
                files = data_dir + filter_name + "/*.1D"
                try:
                    X, Y, covars = preproc.Load_Atlas_Data_Abide(files, 1, 70, sitename=site_name, applythresould=False)
                    mean, varivance = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

                    accuracy[i, j] = mean[:, 2]
                    precision[i, j] = mean[:, 0]
                    recall[i, j] = mean[:, 3]
                    auc[i, j] = mean[:, 5]
                except:
                    pass

            df1 = pd.DataFrame(accuracy, index=atlas_all, columns=site_names)
            df2 = pd.DataFrame(precision, index=atlas_all, columns=site_names)
            df3 = pd.DataFrame(recall, index=atlas_all, columns=site_names)
            df4 = pd.DataFrame(auc, index=atlas_all, columns=site_names)

            filename = classifier_names[0]
            with pd.ExcelWriter('result/Atlas_Site/Cross_{}.xls'.format(filename)) as writer:
                df1.to_excel(writer, sheet_name='Accuracy')
                df2.to_excel(writer, sheet_name='Precision')
                df3.to_excel(writer, sheet_name='Recall')
                df4.to_excel(writer, sheet_name='Auc')
            lastindex=i

        data_dir = "Atlas_Extractor/Nilearn/"
        for i, filter_name in enumerate(nilearn_atlas):
            for j, site_name in enumerate(site_names):
                files = data_dir + filter_name + "/*.npy"
                try:
                    X, Y, covars = preproc.Load_Atlas_Data_NiLearn(files, 1, 100, sitename=site_name, applythresould=False)
                    mean, varivance = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

                    accuracy[i+lastindex, j] = mean[:, 2]
                    precision[i+lastindex, j] = mean[:, 0]
                    recall[i+lastindex, j] = mean[:, 3]
                    auc[i+lastindex, j] = mean[:, 5]
                except:
                    pass

            df1 = pd.DataFrame(accuracy, index=atlas_all, columns=site_names)
            df2 = pd.DataFrame(precision, index=atlas_all, columns=site_names)
            df3 = pd.DataFrame(recall, index=atlas_all, columns=site_names)
            df4 = pd.DataFrame(auc, index=atlas_all, columns=site_names)

            filename = classifier_names[0]
            with pd.ExcelWriter('result/Atlas_Site/Cross_{}.xls'.format(filename)) as writer:
                df1.to_excel(writer, sheet_name='Accuracy')
                df2.to_excel(writer, sheet_name='Precision')
                df3.to_excel(writer, sheet_name='Recall')
                df4.to_excel(writer, sheet_name='Auc')
    def Classify_Wavelet_For_Each_Site1(self):
        preproc = Preprocess()
        site_names = np.unique(phenotype[:, 10])
        # X, Y, df = preproc.Load_Atlas_Data_Abide(files, 1, 70, applythresould=True)

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifier_names = classifier.Get_Classifier_Names()
        filter_bank = ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'bior1.3', 'bior1.5', 'bior2.2',
                       'bior2.4','bior2.6', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior4.4', 'bior5.5', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'sym2', 'sym3', 'sym4',
                       'sym5', 'sym6', 'sym7', 'sym8']

        data_dir = "Atlas_Extractor/WaveletDifumo256/"

        for index, filter_name in enumerate(filter_bank):

            accuracy = np.zeros((len(classifier_names), len(site_names)))
            precision = np.zeros((len(classifier_names), len(site_names)))
            recall = np.zeros((len(classifier_names), len(site_names)))
            auc = np.zeros((len(classifier_names), len(site_names)))

            for index, site_name in enumerate(site_names):
                files = data_dir + filter_name + "/*.npy"
                try:
                    X, Y, covars = preproc.Load_data_wavelet_2d(files, 1, 70, sitename=site_name, applythresould=False)
                    mean, varivance = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

                    accuracy[:, index] = mean[:, 2]
                    precision[:, index] = mean[:, 0]
                    recall[:, index] = mean[:, 3]
                    auc[:, index] = mean[:, 5]
                except:
                    pass

                lastindex = index

            df1 = pd.DataFrame(accuracy, index=classifier_names, columns=site_names)
            df2 = pd.DataFrame(precision, index=classifier_names, columns=site_names)
            df3 = pd.DataFrame(recall, index=classifier_names, columns=site_names)
            df4 = pd.DataFrame(auc, index=classifier_names, columns=site_names)

            filename = os.path.split(filter_name)[1]
            with pd.ExcelWriter('result/Waveletsite/Cross_{}.xls'.format(filename)) as writer:
                df1.to_excel(writer, sheet_name='Accuracy')
                df2.to_excel(writer, sheet_name='Precision')
                df3.to_excel(writer, sheet_name='Recall')
                df4.to_excel(writer, sheet_name='Auc')
    def Classify_Wavelet_For_Each_Site2(self):
        preproc = Preprocess()
        site_names = np.unique(phenotype[:, 10])
        # X, Y, df = preproc.Load_Atlas_Data_Abide(files, 1, 70, applythresould=True)

        pure_classifier = Pure_Classifier()
        classifierlist = pure_classifier.GenerateAllClassifiers()
        for classifier in classifierlist:
            pure_classifier.GenereateClassifier(classifier)
            classifier_names = pure_classifier.Get_Classifier_Names()

            filter_bank = ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'bior1.3', 'bior1.5', 'bior2.2',
                           'bior2.4','bior2.6', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior4.4', 'bior5.5', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'sym2', 'sym3', 'sym4',
                           'sym5', 'sym6', 'sym7', 'sym8']

            #filter_bank= ['bior5.52d(90percent)']
            data_dir = "Wavelet_Extractor/Part10/"

            accuracy = np.zeros((len(filter_bank), len(site_names)))
            precision = np.zeros((len(filter_bank), len(site_names)))
            recall = np.zeros((len(filter_bank), len(site_names)))
            auc = np.zeros((len(filter_bank), len(site_names)))

            misssex = np.empty((len(filter_bank), len(site_names)),dtype=np.object)
            missage = np.empty((len(filter_bank), len(site_names)),dtype=np.object)


            for i, filter_name in enumerate(filter_bank):
                print(filter_bank)
                for j, site_name in enumerate(site_names):
                    files = data_dir + filter_name + "/*.npy"
                    try:
                        X, Y, covars = preproc.Load_data_wavelet_2d(files, 1, 70, sitename=site_name, applythresould=False)
                        mean , varivance,misindex = pure_classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

                        totalcovars =  covars.to_numpy()[:,1:]

                        totalsex = collections.Counter(totalcovars[:,0])
                        totalage = collections.Counter(totalcovars[:,1])

                        miscovars = covars.to_numpy()[misindex,1:]
                        missex = collections.Counter(miscovars[:,0])
                        misage = collections.Counter(miscovars[:,1])

                        for key in totalsex.keys():
                            try:
                                totalsex[key] = missex[key]/totalsex[key]
                            except:
                                totalsex[key]=0

                        for key in totalage.keys():
                            try:
                                totalage[key] = misage[key]/totalage[key]
                            except:
                                totalage[key]=0

                        json_sex = json.dumps(totalsex, indent=4)
                        json_age = json.dumps(totalage, indent=4)

                        accuracy[i, j] = mean[:, 2]
                        precision[i, j] = mean[:, 0]
                        recall[i, j] = mean[:, 3]
                        auc[i, j] = mean[:, 5]

                        misssex[i,j] = json_sex
                        missage[i,j] = json_age

                    except Exception as err:
                        print (err)
                        pass
                df1 = pd.DataFrame(accuracy, index=filter_bank, columns=site_names)
                df2 = pd.DataFrame(precision, index=filter_bank, columns=site_names)
                df3 = pd.DataFrame(recall, index=filter_bank, columns=site_names)
                df4 = pd.DataFrame(auc, index=filter_bank, columns=site_names)
                df5 = pd.DataFrame(misssex, index=filter_bank, columns=site_names)
                df6 = pd.DataFrame(missage, index=filter_bank, columns=site_names)

                filename = classifier_names[0]
                with pd.ExcelWriter('result/Waveletsite/Miss_Cross_Part10_{}.xls'.format(filename)) as writer:
                    df1.to_excel(writer, sheet_name='Accuracy')
                    df2.to_excel(writer, sheet_name='Precision')
                    df3.to_excel(writer, sheet_name='Recall')
                    df4.to_excel(writer, sheet_name='Auc')
                    df5.to_excel(writer, sheet_name='MissSex')
                    df6.to_excel(writer, sheet_name='MisAge')

    def GridSearchForSVC(self):
        preproc =Preprocess()
        classifier = Pure_Classifier()
        path = "Atlas_Extractor/Abide/rois_cc200/*.1d"
        X, Y, covars = preproc.Load_Atlas_Data_Abide(path, 1, 70, applythresould=False)
        classifier.GridSearchKNNClassifier(X,Y)
    def Classify_Atlas_With_DeepLearning(self,resultfilename):
        #This function classify abide autism fmri dataset with 13 classifier methods and save the perfomance of each classifier
        # in the excel file located in the result folder, so before running this code, it is need to create a folder with result name

        preproc =Preprocess()

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifier_names =[ "Deep Learning" ]


        atlas_names = []
        data_dir = "Atlas_Extractor/Abide/*_aal"
        for index,folder in enumerate(glob.glob(data_dir)):
            atlas_names.append(folder.split("\\")[1])

        #data_dir_nilearn = "Atlas_Extractor/Nilearn/*"
        #for index,folder in enumerate(glob.glob(data_dir_nilearn)):
        #    atlas_names.append(folder.split("\\")[1])

        accuracy =np.zeros((len(classifier_names),len(atlas_names)))
        precision =np.zeros((len(classifier_names),len(atlas_names)))
        recall =np.zeros((len(classifier_names),len(atlas_names)))
        auc =np.zeros((len(classifier_names),len(atlas_names)))

        #Train Abide Atlas
        lastindex = 0
        for index,folder in enumerate(glob.glob(data_dir)):
            print (folder)
            files = folder + "/*.1d"
            X, Y, covars = preproc.Load_Atlas_Data_Abide(files, 1, 70, applythresould=False)
            categorical_cols = ['gender', 'age']
            batch_col = 'batch'
            X = np.transpose(X)
            data_combat = neuroCombat(dat=X, covars=covars, batch_col=batch_col, categorical_cols=categorical_cols)["data"]
            X = np.transpose(data_combat)
            mean, varivance = classifier.DoDeepLearning1(X, Y, )

            accuracy[:,index] = mean[:,2]
            precision[:,index] = mean[:,0]
            recall[:,index] = mean[:,3]
            auc[:,index] = mean[:,5]
            lastindex=index

        # Train Nilearn Atlas and save result in the same numpy array accuracy,precision, auc, recall
        #for index, folder in enumerate(glob.glob(data_dir_nilearn)):
        #    print(folder)
        #    files = folder + "/*.npy"
        #    X, Y, df = preproc.Load_Atlas_Data_NiLearn(files, 1, 70,applythresould=True)
        #    mean, varivance = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

#            accuracy[:, index+lastindex] = mean[:, 2]
#            precision[:, index+lastindex] = mean[:, 0]
#            recall[:, index+lastindex] = mean[:, 3]
#            auc[:, index+lastindex] = mean[:, 5]

        df1 = pd.DataFrame(accuracy, index=classifier_names, columns=atlas_names)
        df2 = pd.DataFrame(precision, index=classifier_names, columns=atlas_names)
        df3 = pd.DataFrame(recall, index=classifier_names, columns=atlas_names)
        df4 = pd.DataFrame(auc, index=classifier_names, columns=atlas_names)

        with pd.ExcelWriter('result/Cross_{}.xls'.format(resultfilename)) as writer:
            df1.to_excel(writer, sheet_name='Accuracy')
            df2.to_excel(writer, sheet_name='Precision')
            df3.to_excel(writer, sheet_name='Recall')
            df4.to_excel(writer, sheet_name='Auc')
    def Classify_Wavelet_With_Combat(self,resultfilename):
        preproc =Preprocess()

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifier_names = classifier.Get_Classifier_Names()

        filter_bank = ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'bior1.3', 'bior1.5', 'bior2.2',
                       'bior2.4',
                       'bior2.6', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7',
                       'bior4.4', 'bior5.5', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'sym2', 'sym3', 'sym4',
                       'sym5', 'sym6', 'sym7', 'sym8']

        data_dir = "Atlas_Extractor/WaveletDifumo256/"

        accuracy =np.zeros((len(classifier_names),len(filter_bank)))
        precision =np.zeros((len(classifier_names),len(filter_bank)))
        recall =np.zeros((len(classifier_names),len(filter_bank)))
        auc =np.zeros((len(classifier_names),len(filter_bank)))

        for index,filter_name in enumerate(filter_bank):

            print (filter_name)
            files = data_dir+filter_name+"/*.npy"

            X, Y, covars = preproc.Load_data_wavelet_2d(files, 1, 70,dropcorrelated=False)
            categorical_cols = ['gender', 'age']
            batch_col = 'batch'
            X = np.transpose(X)
            data_combat = neuroCombat(dat=X, covars=covars, batch_col=batch_col, categorical_cols=categorical_cols)["data"]
            X = np.transpose(data_combat)


            mean, varivance = classifier.DoCrossValidation(X, Y, "filename", savetofile=False)

            accuracy[:,index] = mean[:,2]
            precision[:,index] = mean[:,0]
            recall[:,index] = mean[:,3]
            auc[:,index] = mean[:,5]
            lastindex=index


        df1 = pd.DataFrame(accuracy, index=classifier_names, columns=filter_bank)
        df2 = pd.DataFrame(precision, index=classifier_names, columns=filter_bank)
        df3 = pd.DataFrame(recall, index=classifier_names, columns=filter_bank)
        df4 = pd.DataFrame(auc, index=classifier_names, columns=filter_bank)

        with pd.ExcelWriter('result/Cross_{}.xls'.format(resultfilename)) as writer:
            df1.to_excel(writer, sheet_name='Accuracy')
            df2.to_excel(writer, sheet_name='Precision')
            df3.to_excel(writer, sheet_name='Recall')
            df4.to_excel(writer, sheet_name='Auc')

    def Classify_Wavelet_With_Combat_GroupFold(self,resultfilename):
        preproc =Preprocess()

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifier_names = classifier.Get_Classifier_Names()

        filter_bank = ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'bior1.3', 'bior1.5', 'bior2.2',
                       'bior2.4',
                       'bior2.6', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7',
                       'bior4.4', 'bior5.5', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'sym2', 'sym3', 'sym4',
                       'sym5', 'sym6', 'sym7', 'sym8']

        data_dir = "Wavelet_Extractor/Part10/"

        accuracy =np.zeros((len(classifier_names),len(filter_bank)))
        precision =np.zeros((len(classifier_names),len(filter_bank)))
        recall =np.zeros((len(classifier_names),len(filter_bank)))
        auc =np.zeros((len(classifier_names),len(filter_bank)))

        for index,filter_name in enumerate(filter_bank):

            print (filter_name)
            files = data_dir+filter_name+"/*.npy"

            X, Y, covars = preproc.Load_data_wavelet_2d(files, 1, 70,dropcorrelated=False)
            categorical_cols = ['gender', 'age']
            batch_col = 'batch'
            X = np.transpose(X)
            data_combat = neuroCombat(dat=X, covars=covars, batch_col=batch_col, categorical_cols=categorical_cols)["data"]
            X = np.transpose(data_combat)
            groups = covars.to_numpy()
            group = groups[:,0]

            mean, varivance = classifier.DoCrossValidationGroup(X, Y,group, "filename", savetofile=False)

            accuracy[:,index] = mean[:,2]
            precision[:,index] = mean[:,0]
            recall[:,index] = mean[:,3]
            auc[:,index] = mean[:,5]
            lastindex=index


        df1 = pd.DataFrame(accuracy, index=classifier_names, columns=filter_bank)
        df2 = pd.DataFrame(precision, index=classifier_names, columns=filter_bank)
        df3 = pd.DataFrame(recall, index=classifier_names, columns=filter_bank)
        df4 = pd.DataFrame(auc, index=classifier_names, columns=filter_bank)

        with pd.ExcelWriter('result/Cross_{}.xls'.format(resultfilename)) as writer:
            df1.to_excel(writer, sheet_name='Accuracy')
            df2.to_excel(writer, sheet_name='Precision')
            df3.to_excel(writer, sheet_name='Recall')
            df4.to_excel(writer, sheet_name='Auc')
    def Classify_CombileAll_Wavelet(self,resultfilename):

        preproc =Preprocess()

        classifier = Pure_Classifier()
        classifier.GenereateClassifier()
        classifier_names = classifier.Get_Classifier_Names()

        filter_bank = ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'bior1.3', 'bior1.5', 'bior2.2',
                       'bior2.4',
                       'bior2.6', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7',
                       'bior4.4', 'bior5.5', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'sym2', 'sym3', 'sym4',
                       'sym5', 'sym6', 'sym7', 'sym8']

        data_dir = "Wavelet_Extractor/Part10/"

        accuracy =np.zeros( (len(classifier_names),len(filter_bank)))
        precision =np.zeros((len(classifier_names),len(filter_bank)))
        recall =np.zeros(   (len(classifier_names),len(filter_bank)))
        auc =np.zeros(      (len(classifier_names),len(filter_bank)))

        All_X = []
        for index,filter_name in enumerate(filter_bank):
            print (filter_name)
            files = data_dir+filter_name+"/*.npy"
            X, Y, covars = preproc.Load_data_wavelet_2d(files, 1, 70,dropcorrelated=False)
            if (index==0):
                All_X = X
            else:
                All_X=np.concatenate((All_X,X),axis=1)
        categorical_cols = ['gender', 'age']
        batch_col = 'batch'

        X = np.transpose(All_X)
        data_combat = neuroCombat(dat=X, covars=covars, batch_col=batch_col, categorical_cols=categorical_cols)["data"]
        X = np.transpose(data_combat)
        groups = covars.to_numpy()
        group = groups[:,0]

        mean, varivance = classifier.DoCrossValidationGroup(X, Y,group, "filename", savetofile=False)

        accuracy[:,index] = mean[:,2]
        precision[:,index] = mean[:,0]
        recall[:,index] = mean[:,3]
        auc[:,index] = mean[:,5]


        df1 = pd.DataFrame(accuracy, index=classifier_names)
        df2 = pd.DataFrame(precision, index=classifier_names)
        df3 = pd.DataFrame(recall, index=classifier_names)
        df4 = pd.DataFrame(auc, index=classifier_names)

        with pd.ExcelWriter('result/Cross_{}.xls'.format(resultfilename)) as writer:
            df1.to_excel(writer, sheet_name='Accuracy')
            df2.to_excel(writer, sheet_name='Precision')
            df3.to_excel(writer, sheet_name='Recall')
            df4.to_excel(writer, sheet_name='Auc')

    def GridSearchForRandomForest(self):
        preproc = Preprocess()
        classifier = Pure_Classifier()
        path = "Wavelet_Extractor/bior2.6/*.npy"
        X, Y, covars = preproc.Load_data_wavelet_2d(path, 1, 70, applythresould=False)
        categorical_cols = ['gender', 'age', 'disease']
        batch_col = 'batch'
        X = np.transpose(X)
        data_combat = neuroCombat(dat=X, covars=covars, batch_col=batch_col, categorical_cols=categorical_cols)["data"]
        X = np.transpose(data_combat)
        classifier.GridSearchRandomForest(X, Y)

#Run MOdels

autism = AutismClassifiers()
#autism.Classify_Atlas("AllAge 1-10",1,10)
#autism.Classify_Atlas("AllAge 11-20",11,20)
#autism.Classify_Atlas("AllAge 20-50",20,60)

#autism.Classify_Atlas_With_Combat("Combat 1-10",1,10)
#autism.Classify_Atlas_GroupKFold("Atlas",0,100)
#autism.Classify_Atlas_With_Combat("Combat 20-50",20,60)

#autism.Classify_Atlas_For_Each_Site()
#autism.Classify_Atlas_With_Combat("Atlas_Combat")

#autism.Classify_Atlas_With_DeepLearning("DeepWithCombat")
#autism.Classify_Atlas_For_Each_Site("Atlas_Site")
autism.Classify_Wavelet_For_Each_Site2()
#autism.Classify_Atlas_Nilearn_For_Each_Site()
#autism.Classify_Wavelet_With_Combat_GroupFold("Wave_Part10")
#autism.Classify_CombileAll_Wavelet("Wave_Combine_Part5")
#autism.GridSearchForRandomForest()