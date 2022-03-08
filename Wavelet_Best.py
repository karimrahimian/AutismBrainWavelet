import pandas as pd
import glob
import os
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

phenotype = pd.read_excel('Labels.xls')
phenotype=phenotype.to_numpy()
site_names = np.unique(phenotype[:, 10])

xls = pd.read_excel("Result/split5_gaussianNB.xls",sheet_name="Accuracy")
atlas_names = xls.to_numpy()[:,0]

accuracy_max =np.empty((len(site_names)),dtype=np.float)
precision_max =np.empty((len(site_names)),dtype=np.float)
auc_max =np.empty((len(site_names)),dtype=np.float)
recall_max =np.empty((len(site_names)),dtype=np.float)
specificity_max =np.empty((len(site_names)),dtype=np.float)

accuracy_index =np.empty((len(site_names)),dtype=np.int)
classifier_names = []
Sheets = ["Accuracy", "Precision", "Auc", "Recall", "Specification"]
sheet = Sheets[0]
xlsfile = glob.glob("Result/*.xls")

for i,file in enumerate(xlsfile):
    filename = os.path.basename(file)
    classifier = filename[6:len(filename)-4].replace("Classifier","")
    print(classifier)
    xls = pd.read_excel(file,sheet_name="Accuracy")
    data = xls.to_numpy()[:,1:]

    xls = pd.read_excel(file,sheet_name="Precision")
    precisiondata = xls.to_numpy()[:,1:]

    xls = pd.read_excel(file,sheet_name="Recall")
    recalldata = xls.to_numpy()[:,1:]

    xls = pd.read_excel(file,sheet_name="Auc")
    aucdata = xls.to_numpy()[:,1:]

    xls = pd.read_excel(file,sheet_name="Specification")
    specificitydata = xls.to_numpy()[:,1:]

    if (i==0):
        accuracy_max = np.max(data,axis=0)
        accuracy_index = np.argmax(data,axis=0)

        for j in range(len(site_names)):
            classifier_names.append(classifier)
            rowindex = accuracy_index[j]

            precision_max[j] = precisiondata[rowindex,j]
            recall_max[j] = recalldata[rowindex,j]
            auc_max[j] = aucdata[rowindex,j]
            specificity_max[j] = specificitydata[rowindex,j]
    else:
        accuracy = np.max(data,axis=0)
        acc_index = np.argmax(data,axis=0)
        for j in range(len(accuracy_max)):
            if (accuracy[j]>accuracy_max[j]):
                accuracy_max[j] = accuracy[j]
                accuracy_index[j] = acc_index[j]
                classifier_names[j] = classifier

                rowindex = accuracy_index[j]
                precision_max[j] = precisiondata[rowindex, j]
                recall_max[j] = recalldata[rowindex, j]
                auc_max[j] = aucdata[rowindex, j]
                specificity_max[j] = specificitydata[rowindex, j]

join_names = []

for i in range(len(site_names)):
    new_name = site_names[i] + "_"+ atlas_names[ accuracy_index[i]] +"_"+ classifier_names[i]
    join_names.append(new_name)
    print("{},{},{},{}\n".format(site_names[i], accuracy_max[i],atlas_names[ accuracy_index[i]],classifier_names[i]))
finaldata = pd.DataFrame({"Sitename":site_names,
                          "AtlasName":atlas_names[ accuracy_index],
                          "Classifiers":classifier_names,
                          "join_Name":join_names,
                          "Accuracy":accuracy_max,
                          "Precision":precision_max,
                          "Recall":recall_max,
                          "AUC":auc_max,
                          "Specificity":specificity_max })
finaldata.to_excel("Wavelet_Total.xls")

