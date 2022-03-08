import pandas as pd
import numpy as np
import glob
import os
phenotype = pd.read_excel('Labels.xls')
phenotype=phenotype.to_numpy()
site_names = np.unique(phenotype[:, 10])
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
rng = np.random.default_rng(123)
import json

def add_labels(angles, values, labels, offset, ax):
    padding = 4
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
        rotation, alignment = get_label_rotation(angle, offset)
        ax.text(
            x=angle,
            y=value + padding,
            s=label,
            ha=alignment,
            va="center",
            rotation=rotation,
            rotation_mode="anchor"
        )
def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else:
        alignment = "left"
    return rotation, alignment
def DrawCircular(df,Group_Name,GROUPS_SIZE,filename):
    df.head(3)

    #GROUPS_SIZE = [5,3,4,8,2,20]
    #Group_Name = ["A", "B", "C", "D","E","F"]

    VALUES = df["value"].values
    LABELS = df["name"].values
    WIDTH = 2 * np.pi / (len(VALUES))
    OFFSET = np.pi / 2

    VALUES = df["value"].values
    GROUP = df["group"].values

    PAD = 3
    ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))+10
    ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)

    offset = 0
    IDXS = []

    for size in GROUPS_SIZE:
        IDXS += list(range(offset + PAD, offset + size + PAD))
        offset += size + PAD

    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(OFFSET)
    ax.set_ylim(-100, 100)
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    COLORS = [f"C{i}" for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]

    ax.bar(
        ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS,
        edgecolor="white", linewidth=2
    )
    add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)
    offset = 0
    for group, size in zip(Group_Name, GROUPS_SIZE):
        x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
        ax.plot(x1, [-5] * 50, color="#333333")

        ax.text(
            np.mean(x1), -20, group, color="#333333", fontsize=14,
            fontweight="bold", ha="center", va="center"
        )

        x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
        ax.plot(x2, [20] * 50, color="#bebebe", lw=0.3)
        ax.plot(x2, [40] * 50, color="#bebebe", lw=0.3)
        ax.plot(x2, [60] * 50, color="#bebebe", lw=0.8)
        ax.plot(x2, [80] * 50, color="#bebebe", lw=0.8)
        offset += size + PAD
    plt.savefig("result/{}.png".format(filename))

path_result = "Result/*.*"

accuracy =np.empty((11,len(site_names)),dtype=np.object)
auc =np.empty((11,len(site_names)),dtype=np.object)
precision =np.empty((11,len(site_names)),dtype=np.object)
recall =np.empty((11,len(site_names)),dtype=np.object)
bestfilterbank =np.empty((11,len(site_names)),dtype=np.object)
missex = np.empty((11,len(site_names)),dtype=np.object)
misage = np.empty((11,len(site_names)),dtype=np.object)

classifiers = []

def findbest(maxindex,param):
    data = pd.read_excel(file,sheet_name=param).to_numpy()
    filterbank = data[:,0]
    data=data[:,1:]
    bestvalue = []
    bestfilter = []
    for j,index in enumerate(maxindex):
        bestvalue.append(data[index,j])
        bestfilter.append( filterbank[index] )
        
    return bestvalue, bestfilter
def findbestfinal(maxindex,data,filterbank):
    bestvalue = []
    bestfilter = []
    for j,index in enumerate(maxindex):
        bestvalue.append(data[index,j])
        bestfilter.append( filterbank[index,j] )
    return bestvalue, bestfilter

for i,file in enumerate(glob.glob(path_result)):
    accdata = pd.read_excel(file,sheet_name='Accuracy').to_numpy()
    filter_bank = accdata[:,0]
    accdata = accdata[:,1:]
    filename = os.path.basename(file)
    classifiers.append(filename[11:])

    maxindex = np.argmax(accdata,axis=0)
    maxvalue = np.max(accdata,axis=0)

    accuracy[i,:] = maxvalue

    auc [i,:],bestfilterbank[i,:]  = findbest(maxindex,'Auc')
    precision[i,:],dontuse = findbest(maxindex,'Precision')
    recall [i,:],dontuse = findbest(maxindex,'Recall')
    misage [i,:],dontuse = findbest(maxindex,'MisAge')
    missex [i,:],dontuse = findbest(maxindex,'MissSex')

    #print(aucdata[maxindex,:])

plt.figure(figsize=(20,10),dpi=800)
plt.subplot(221,)
plt.boxplot(accuracy, showmeans=True,labels=site_names )
#plt.xlabel('Sitename', fontsize=18)
plt.ylabel('Accuracy', fontsize=16)
plt.xticks(rotation=90)

plt.subplot(222,)
plt.boxplot(auc, showmeans=True,labels=site_names )
#plt.xlabel('Sitename', fontsize=18)
plt.ylabel('Auc', fontsize=16)
plt.xticks(rotation=90)

plt.subplot(223,)
plt.boxplot(precision, showmeans=True, labels=site_names)
plt.xlabel('Sitename', fontsize=18)
plt.ylabel('Precision', fontsize=16)
plt.xticks(rotation=90)

plt.subplot(224,)
plt.boxplot(recall, showmeans=True, labels=site_names)
plt.xlabel('Sitename', fontsize=18)
plt.ylabel('Recall', fontsize=16)
plt.xticks(rotation=90)

plt.savefig("result/box2.png",bbox_inches='tight')


maxindex = np.argmax(accuracy,axis=0)
maxvalue = np.max(accuracy,axis=0)


#***********************************************  find misclassify bestage and bestsex
bestsex = []
bestage = []
for i in range(missex.shape[1]):
    bestsex.append(missex[maxindex[i], i])
    bestage.append(misage[maxindex[i], i])

#*********************************************** plot misclassify age
dfage = []
Group_Size = []
for i,item in enumerate(bestage):
    temp = json.loads(item)
    size = 0
    for key in temp.keys():
        dfage.append([key,float(temp[key]),site_names[i]])
        size+=1
    Group_Size.append(size)

dfage = np.array(dfage)
values = dfage[:,1].astype(np.float)
dfage = pd.DataFrame({"name":dfage[:,0], "value":values*100 , "group":dfage[:,2] })

DrawCircular(dfage,site_names,Group_Size,"CircularAge")

#*********************************************** plot misclassify sex
dfsex = []
Group_Size = []
for i,item in enumerate(bestsex):
    temp = json.loads(item)
    size = 0
    for key in temp.keys():
        dfsex.append([key,float(temp[key]),site_names[i]])
        size+=1
    Group_Size.append(size)

dfsex = np.array(dfsex)
values = dfsex[:,1].astype(np.float)
dfsex = pd.DataFrame({"name":dfsex[:,0], "value":values*100 , "group":dfsex[:,2] })

DrawCircular(dfsex,site_names,Group_Size,"CircularSex")

accuracy_total = maxvalue
auc_final,filterfinal = findbestfinal(maxindex,auc,bestfilterbank)
precision_final,filterfinal = findbestfinal(maxindex,precision,bestfilterbank)
recall_final,filterfinal = findbestfinal(maxindex,recall,bestfilterbank)

#del maxindex,maxvalue,auc,accuracy,precision,i,file,recall

