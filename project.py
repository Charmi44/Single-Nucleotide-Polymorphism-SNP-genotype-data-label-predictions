#Team members :
#1. Charmi Patel - cp423
#2. Prithwish Ganuly - pg422

import sys
import array
import copy

#Importing required sklearn modules and creating alias
from sklearn import linear_model as lm
from sklearn.naive_bayes import GaussianNB as g_nb
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid as NC
from sklearn.model_selection import train_test_split as tts



def mergeColumn(a, b):
    return [x + y for x, y in zip(a, b)]


def extractColumn(matrix, i):
    return [[row[i]] for row in matrix]


def dataCreation(features, dat):
    updateData = extractColumn(dat, features[0])
    updateLabel = array.array("i")
    features.remove(features[0])
    length = len(features)
    for i in range(0, length, 1):
        temp = extractColumn(dat, features[0])
        updateData = mergeColumn(updateData, temp)
        features.remove(features[0])
    return updateData


def PearsonCorrtin(x, y, fi):
    sum_x1 = 0
    sum_x2 = 0
    ro = len(x)
    co = len(x[0])
    switch = 0
    pc = array.array("f")
    for i in range(0, co, 1):
        switch += 1
        sum_y1 = 0
        sum_y2 = 0
        sum_xy = 0
        for j in range(0, ro, 1):
            if (switch == 1):
                sum_x1 += y[j]
                sum_x2 += y[j] ** 2
            sum_y1 += x[j][i]
            sum_y2 += x[j][i] ** 2
            sum_xy += y[j] * x[j][i]
        r = (ro * sum_xy - sum_x1 * sum_y1) / ((ro * sum_x2 - (sum_x1 ** 2)) * (ro * sum_y2 - (sum_y1 ** 2))) ** (0.5)
        pc.append(abs(r))

    print_it = array.array("f")
    features_extracted = array.array("i")
    for i in range(0, fi, 1):
        selected_features = max(pc)
        print_it.append(selected_features)
        featureIndex = pc.index(selected_features)
        pc[featureIndex] = -1
        features_extracted.append(featureIndex)
    return features_extracted

#Reading data
datafile = sys.argv[1]
data = []
print("Reading data and training Lables")
with open(datafile, "r") as infile:
    for line in infile:
        temp = line.split()
        l = array.array("i")
        for i in temp:
            l.append(int(i))
        data.append(l)

labelfile = sys.argv[2]
trainlabels = array.array("i")
with open(labelfile, "r") as infile:
    for line in infile:
        temp = line.split()
        trainlabels.append(int(temp[0]))

print("Data reading completed ", end="")

features = 10
rows = len(data)
cols = len(data[0])
rowsl = len(trainlabels)


# Performing dimensionality Reduction on extracted data
print("Starting dimensionality Reduction...")
requied_features = PearsonCorrtin(data, trainlabels, 2000)

print("  Dimensionality Reduction Done ", end="")

stored_features = copy.deepcopy(requied_features)

data1 = dataCreation(requied_features, data)


classify_svm = svm.SVC(gamma=0.001)
classify_log = lm.LogisticRegression()
classify_gnb = g_nb()
classify_nc = NC()

overallAccuracy = array.array("f")
allFeatures = []

SVMAccuracycuracy = 0
accuracy_score = 0
logAccuracy = 0
gnbAccuracy = 0
ncAccuracy = 0

myAccuracy = 0

iterations = 5
print(" Iterating cross validation : ", end="")
for i in range(iterations):

    print(i)

    X_train, X_test, y_train, y_test = tts(
        data1, trainlabels, test_size=0.3)

    newRows = len(X_train)
    newCols = len(X_train[0])
    newRowst = len(X_test)
    newColst = len(X_test[0])

    newRowsL = len(y_train)


    PearFeatures = PearsonCorrtin(X_train, y_train, features)

    allFeatures.append(PearFeatures)
    argument = copy.deepcopy(PearFeatures)

    data_fea = dataCreation(argument, X_train)

    classify_svm.fit(data_fea, y_train)
    classify_log.fit(data_fea, y_train)
    classify_gnb.fit(data_fea, y_train)
    classify_nc.fit(data_fea, y_train)

    TestFeatures = PearsonCorrtin(X_test, y_test, features)

    test_features = dataCreation(TestFeatures, X_test)

    total_features = len(test_features)
    svm_count = 0
    log_count = 0
    gnb_count = 0
    nc_count = 0
    my_counter = 0

    for j in range(0, total_features, 1):
        svm_pred = int(classify_svm.predict([test_features[j]]))
        log_pred = int(classify_log.predict([test_features[j]]))
        gnb_pred = int(classify_gnb.predict([test_features[j]]))
        nc_pred = int(classify_nc.predict([test_features[j]]))
        h = svm_pred + log_pred + gnb_pred + nc_pred
        if (h >= 3):
            myPredlabel = 1
        elif (h <= 1):
            myPredlabel = 0
        else:
            myPredlabel = svm_pred
        if (myPredlabel == y_test[j]):
            my_counter += 1
        if (svm_pred == y_test[j]):
            svm_count += 1
        if (log_pred == y_test[j]):
            log_count += 1
        if (gnb_pred == y_test[j]):
            gnb_count += 1
        if (nc_pred == y_test[j]):
            nc_count += 1


    SVMAccuracycuracy += svm_count / total_features
    logAccuracy += log_count / total_features

    gnbAccuracy += gnb_count / total_features
    ncAccuracy += nc_count / total_features

    myAccuracy += my_counter / total_features
    overallAccuracy.append(my_counter / total_features)


print(" Done", end="")

bestAccuracy = max(overallAccuracy)
bestIndex = overallAccuracy.index(bestAccuracy)
bestFeatures = allFeatures[bestIndex]

print("\nFeatures: ", features)

originalFeatures = array.array("i")
for i in range(0, features, 1):
    real_Index = stored_features[bestFeatures[i]]
    originalFeatures.append(real_Index)

print("The features are: ", originalFeatures)

arg1 = copy.deepcopy(originalFeatures)
AccurateData = dataCreation(arg1, data)


classify_svm.fit(AccurateData, trainlabels)
classify_log.fit(AccurateData, trainlabels)
classify_gnb.fit(AccurateData, trainlabels)
classify_nc.fit(AccurateData, trainlabels)

svm_counter = 0
counter_l = 0
k = len(AccurateData)
for i in range(0, k, 1):
    svm_pred = int(classify_svm.predict([AccurateData[i]]))
    log_pred = int(classify_log.predict([AccurateData[i]]))
    gnb_pred = int(classify_gnb.predict([AccurateData[i]]))
    nc_pred = int(classify_nc.predict([AccurateData[i]]))
    h = svm_pred + log_pred + gnb_pred + nc_pred
    if (h >= 3):
        myPredlabel = 1
    elif (h <= 1):
        myPredlabel = 0
    else:
        myPredlabel = svm_pred
    if (myPredlabel == trainlabels[i]):
        counter_l += 1
    if (svm_pred == trainlabels[i]):
        svm_counter += 1

FinalAccuracy = counter_l / k
SVMAccuracy = svm_counter / k
print("The Accuracy is: ", FinalAccuracy * 100)


print("\nPredicted labels of the test data are stored in testLabels file")

testfile = sys.argv[3]
testdata = []
print("Reading stored test data...")
with open(testfile, "r") as infile:
    for line in infile:
        temp = line.split()
        l = array.array("i")
        for i in temp:
            l.append(int(i))
        testdata.append(l)

print("  Test data reading completed")

print("   Feature Extraction started...")

arg2 = copy.deepcopy(originalFeatures)
testdata1 = dataCreation(arg2, testdata)


print("   Feature Extraction ended")

file1 = open("testLabels", "w+")

for i in range(0, len(testdata1), 1):
    label1 = int(classify_svm.predict([testdata1[i]]))
    label2 = int(classify_log.predict([testdata1[i]]))
    label3 = int(classify_gnb.predict([testdata1[i]]))
    label4 = int(classify_nc.predict([testdata1[i]]))
    h = label1 + label2 + label3 + label4
    if (h >= 3):
        file1.write(str(1) + " " + str(i) + "\n")
    elif (h <= 1):
        file1.write(str(0) + " " + str(i) + "\n")
    else:
        file1.write(str(label1) + " " + str(i) + "\n")

print(" Done ")
