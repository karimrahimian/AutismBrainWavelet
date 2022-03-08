from scipy.stats import norm
import numpy as np
import glob
import os
from sklearn.preprocessing import RobustScaler,StandardScaler,QuantileTransformer,MinMaxScaler
import math
import pandas as pd

import os

from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, roc_auc_score,confusion_matrix
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

from  sklearn.utils.multiclass import  type_of_target
from scipy import stats

from sklearn.model_selection import StratifiedKFold


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
        self.kfold = 10
        self.classifier = []
        pass
    def logit_p1value(self,model, x):
        p1 = model.predict_proba(x)
        n1 = len(p1)
        m1 = len(model.coef_[0]) + 1
        coefs = np.concatenate([model.intercept_, model.coef_[0]])
        x_full = np.matrix(np.insert(np.array(x), 0, 1, axis=1))
        answ = np.zeros((m1, m1))
        for i in range(n1):
            answ = answ + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p1[i, 1] * p1[i, 0]
        vcov = np.linalg.inv(np.matrix(answ))
        se = np.sqrt(np.diag(vcov))
        t1 = coefs / se
        p1 = (1 - norm.cdf(abs(t1))) * 2
        return p1
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
        #voting = VotingClassifier(estimators=[('SVM', svclassifier), ('DT', dt), ('LR', lr)], voting='hard')
        #histogramgradient = HistGradientBoostingClassifier(min_samples_leaf=1,   max_depth = 2,  learning_rate = 1,max_iter = 1)
        self.classifier = [knn,nivebase,dt,lr,svclassifier,randomforest,ridge,mlp,ada,bagging,sgdclassifier]
        #self.classifier = [knn]
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
        evlP = [[0 for x in range(7)] for YY in range(self.kfold)]
        k = 0
        misindex = []
        for train_index, test_index in kf.split(Data,y):
            classifier.fit(Data[train_index], y[train_index])
            y_pred = classifier.predict(Data[test_index])
            y_test = y[test_index]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp)


            evlP[k][0] = (precision_score(y_test, y_pred, average='macro',labels=np.unique(y_pred)))
            evlP[k][1] = (f1_score(y_test, y_pred, average='macro',labels=np.unique(y_pred)))
            evlP[k][2] = (accuracy_score(y_test, y_pred))
            evlP[k][3] = (recall_score(y_test, y_pred, average="weighted"))
            evlP[k][4] = (matthews_corrcoef(y_test, y_pred))
            evlP[k][5] = self.multiclass_roc_auc_score(y_test, y_pred)
            evlP[k][6] = specificity

            index_counter = 0
            for index in test_index:
                if (y_pred[index_counter]!=y[index]):
                    misindex.append(index)
                index_counter+=1

            k += 1

        average = np.matrix(evlP)
        average = average.mean(axis=0)
        return np.array(average[0]), misindex
    def __Cross_Validation_Group(self, X, y,groups, classifier, kf):
        evlP = [[0 for x in range(7)] for YY in range(self.kfold)]
        k = 0
        data1, data2, data3, data4 = self.__GroupCrossFoldKarim(X=X,Y=y,group=groups,nfold= self.kfold)
        for trainX, trainY, testX, testY in zip(data1, data2, data3, data4):
            classifier.fit(trainX, trainY)
            y_pred = classifier.predict(testX)
            y_test = testY

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp)

            evlP[k][0] = (precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred)))
            evlP[k][1] = (f1_score(y_test, y_pred, average='macro', labels=np.unique(y_pred)))
            evlP[k][2] = (accuracy_score(y_test, y_pred))
            evlP[k][3] = (recall_score(y_test, y_pred, average="weighted"))
            evlP[k][4] = (matthews_corrcoef(y_test, y_pred))
            evlP[k][5] = self.multiclass_roc_auc_score(y_test, y_pred)
            evlP[k][6] = specificity
            k += 1

        average = np.matrix(evlP)
        average = average.mean(axis=0)
        return np.array(average[0])
    def __TrainAndTest(self, xTrain, yTrain, xTest, yTest, classifier):
        evlP = np.zeros(7)
        classifier.fit(xTrain, yTrain)
        y_pred = classifier.predict(xTest)
        y_test = yTest
        tn, fp, fn, tp = confusion_matrix(y_test, y_test).ravel()
        specificity = tn / (tn + fp)

        evlP[0] = (precision_score(y_test, y_pred, average='micro'))
        evlP[1] = (f1_score(y_test, y_pred, average='macro'))
        evlP[2] = (accuracy_score(y_test, y_pred))
        evlP[3] = (recall_score(y_test, y_pred, average="weighted"))
        evlP[4] = (matthews_corrcoef(y_test, y_pred))
        evlP[5] = self.multiclass_roc_auc_score(y_test, y_pred)
        evlP[6] = specificity

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
            for j in range(i):
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
    def Classify_Wavelet_For_Each_Site(self,waveletdir, split):
        preproc = Preprocess()
        site_names = np.unique(phenotype[:, 10])
        # X, Y, df = preproc.Load_Atlas_Data_Abide(files, 1, 70, applythresould=True)

        pure_classifier = Pure_Classifier()
        pure_classifier.kfold = 5
        classifierlist = pure_classifier.GenerateAllClassifiers()
        for classifier in classifierlist:
            pure_classifier.GenereateClassifier(classifier)
            classifier_names = pure_classifier.Get_Classifier_Names()

            filter_bank = ['db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'bior1.3', 'bior1.5', 'bior2.2',
                           'bior2.4','bior2.6', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior4.4', 'bior5.5', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'sym2', 'sym3', 'sym4',
                           'sym5', 'sym6', 'sym7', 'sym8']

            #filter_bank= ['bior5.52d(90percent)']
            data_dir = waveletdir

            accuracy = np.zeros((len(filter_bank), len(site_names)))
            precision = np.zeros((len(filter_bank), len(site_names)))
            recall = np.zeros((len(filter_bank), len(site_names)))
            auc = np.zeros((len(filter_bank), len(site_names)))
            specification = np.zeros((len(filter_bank), len(site_names)))

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
                        specification[i, j] = mean[:, 6]

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
                df7 = pd.DataFrame(specification, index=filter_bank, columns=site_names)

                filename = classifier_names[0]
                with pd.ExcelWriter('result/Split{}_{}.xls'.format(split,filename)) as writer:
                    df1.to_excel(writer, sheet_name='Accuracy')
                    df2.to_excel(writer, sheet_name='Precision')
                    df3.to_excel(writer, sheet_name='Recall')
                    df4.to_excel(writer, sheet_name='Auc')
                    df5.to_excel(writer, sheet_name='MissSex')
                    df6.to_excel(writer, sheet_name='MisAge')
                    df7.to_excel(writer, sheet_name='Specification')
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



autism = AutismClassifiers()
autism.Classify_Wavelet_For_Each_Site( waveletdir="Wavelet_Extractor/Part10/", split = 10)

