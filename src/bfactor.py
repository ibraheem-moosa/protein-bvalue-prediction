import os
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score


directory = "/home/ahsan/Desktop/MS Thesis/bfactor/output/" #appropriate location of output folder
directory = './data/'

amino_acids=dict(CYS = 0,
TRP = 1,
MET = 2,
HIS = 3,
GLN = 4,
TYR = 5,
PHE = 6,
ASN = 7,
PRO = 8,
ARG = 9,
ILE = 10,
THR = 11,
LYS = 12,
ASP = 13,
SER = 14,
GLU = 15,
VAL = 16,
GLY = 17,
ALA = 18,
LEU = 19,
other = 20
)




def prepareDataSet():

    SAMPLE_INTERVAL = 20
    MAX_FILE_COUNT = 50

    FEATURE_DIMENSION = 63

    X = np.empty((0,FEATURE_DIMENSION), float)
    y = np.empty((0), float)

    fileCount = 0

    for fileName in os.listdir(directory):
        fileCount += 1;

        if(fileCount==MAX_FILE_COUNT):
            break
        # print(fileName)
        input = np.genfromtxt(directory+fileName, dtype= None, encoding=None)

        totalAcids, totalFreq = calculateFrequency(input)

        # print(totalFreq, totalAcids)

        freq=dict((acid, 0) for acid in amino_acids.keys())
        sampleCount = 0;

        for(acid, bValue) in input:
            if(acid not in amino_acids):
                acid = "other"
            if(acid not in freq):
                freq[acid] = 0
            freq[acid] += 1

            sampleCount += 1
            if(sampleCount % SAMPLE_INTERVAL == 0):
                sampleCount = 0
                featureVector = extractFeature(freq, totalFreq, totalAcids, acid)

                # print(featureVector)


                X = np.append(X, featureVector, axis=0)

                y = np.append(y, bValue)
        # print(X)
        # print(y)

    return X, y



def calculateFrequency(input):
        freq=dict((acid, 0) for acid in amino_acids.keys())
        totalAcids = 0;
        for(acid, bValue) in input:
            if(acid not in amino_acids):
                acid = "other"
            if(acid not in freq):
                freq[acid] = 0
            freq[acid] += 1

            totalAcids += 1

        return totalAcids, freq;



def extractFeature(freq, totalFreq, totalAcids, acid_name):
    featureVector = [(freq[acid]*1.0/totalAcids) for acid in amino_acids.keys()]


    featureVector.extend([totalFreq[acid]*1.0/totalAcids for acid in amino_acids.keys()])


    if acid_name in amino_acids:
        x = np.array([acid for acid in amino_acids.keys()]) == acid_name
    else:
        x = np.array([acid for acid in amino_acids.keys()]) == "other"

    # print(acid_name, x)

    featureVector.extend(x)

    return np.array([featureVector])




def main():
    X, y = prepareDataSet()

    print(len(y))


    clf = SVR(C=1.0, epsilon=0.2)

    # clf.fit(X[:20], y[:20])

    scores = cross_val_score(clf, X, y, cv=10)
    print(scores)

if __name__ == "__main__":
    main()
