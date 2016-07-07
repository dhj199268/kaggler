from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
# from scipy import fftpack


def PCAProcess(data, components):
    estimater = decomposition.SparsePCA(n_components=components, n_jobs=3)
    pcadata = estimater.fit_transform(data)
    return pcadata, estimater


def TrainData(data, label):
    estimater = RandomForestClassifier(n_estimators=100, n_jobs=3, bootstrap=True)
    # estimater=KNeighborsClassifier(n_neighbors=10)
    estimater.fit(data, label)
    return estimater


def AddFeatures(data):
    mean = np.mean(a=data, axis=1)
    std = np.mean(a=data, axis=1)
    # max = np.min(data, axis=1)
    # min = np.max(data, axis=1)

    # fft = fftpack.fft(data)
    # fft = np.abs(fft)
    # data = np.c_[data, mean]
    # data = np.c_[data, std]
    # data = np.c_[data, fft]
    data = np.column_stack((data, mean, std))
    # data=np.column_stack((data,))
    return data


if __name__ == '__main__':
    filename = r"H:\Kaggle\data\Digit Recognition\train.csv"
    modelfile = r"H:\Kaggle\data\Digit Recognition\model.pkl"
    testfilename = r"H:\Kaggle\data\Digit Recognition\test.csv"
    savefile = r"H:\Kaggle\data\Digit Recognition\result.csv"
    print("load data")
    data = pd.read_csv(filename)
    labeldata = data[data.columns[0]].values
    traindata = data[data.columns[1:]].values
    traindata = AddFeatures(traindata)
    print "training model"
    model = TrainData(traindata, labeldata)
    print "training model sucessfull,save model"
    joblib.dump(model, filename=modelfile, compress=3)
    # model = joblib.load(modelfile)
    print "predict data"
    testdata = pd.read_csv(testfilename)
    testdata = AddFeatures(testdata)
    index = testdata.shape[0]
    result =model.predict_proba(testdata)
    # result = model.predict(testdata)
    print "save result data"
    result = result.T
    resutldata = pd.DataFrame(result, index=np.arange(1, index + 1), columns=['Label'])
    resutldata.to_csv(savefile)
    # np.save(savefile,result)
