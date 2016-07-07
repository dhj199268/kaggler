import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn import mixture
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

def SetGender(gender):
    if gender == "male":
        return 1
    elif gender == 'female':
        return 0


def SetEmbarked(embark):
    # print embark
    if embark == "S":
        return 0
    elif embark == "Q":
        return 1
    elif embark == "C":
        return 2
        # elif embark =='nan':
        #     return 3


def SetTicket(ticeket):
    # ticeket = filter(str.isdigit, ticeket)
    # # filter(str.isdigit, ticeket)
    # # print ticeket
    # if ticeket == '':
    #     return 0
    # ticeket = int(ticeket)
    # # print type(ticeket)
    # return ticeket
    if 'A' in ticeket:
        return 0
    elif 'SOTON' in ticeket:
        return 1
    elif 'W' in ticeket:
        return 2
    elif 'PC 17' in ticeket:
        return 3
    else:
        return 4


def SetAge(x):
    if x < 1:
        return True


class Titanic(object):
    srcdata = None
    traindata = None
    labldata = None
    testdata = None
    resultdata = None
    estimater = None
    agemedien = None
    faremedien = None
    colist = ("PassengerId", "Name", "Cabin")

    def __init__(self):
        pass

    def LoadTrainData(self, filename):
        print "load train data"
        self.srcdata = pd.read_csv(filename)
        self.labldata = self.srcdata["Survived"].values.copy()
        del self.srcdata["Survived"]

    def LoadTestData(self, filename):
        print "load test data"
        data = pd.read_csv(filename)
        self.testdata = self.Proessing(data)

    # def SavaResult(self, filename):
    #     print "load test data"
    #     self.resultdata.to_csv(filename)

    def __SetData(self, data):
        print"set features"
        for col in self.colist:
            del data[col]
        return data

    def Proessing(self, data):
        # print "processing data"
        # self.__SetData(data=data)
        print"processing gender"
        data["Sex"] = data["Sex"].map(SetGender)
        print "process age"
        if self.agemedien == None:
            self.agemedien = np.median(data["Age"])
        # median = int(median)
        data["Age"].fillna(self.agemedien, inplace=True)
        print"process Embarked"
        data["Embarked"] = data["Embarked"].fillna(method='bfill')
        data["Embarked"] = data["Embarked"].map(SetEmbarked)

        print"process ticket"
        # data["Ticket"]=data["Ticket"].fillna(method='bfill')
        data["Ticket"] = data["Ticket"].map(SetTicket)

        "proecss fare"
        if self.faremedien == None:
            self.faremedien = np.mean(data["Fare"])
        data["Fare"] = data["Fare"].fillna(self.faremedien)

        # "proecess famliy"
        # data["Family"] = data["SibSp"] + data["Parch"]
        # del data["SibSp"]
        # del data["Parch"]

    @staticmethod
    def SaveData(filname, data):
        data.to_csv(filname)

    def Run(self):
        self.traindata = self.__SetData(self.srcdata)
        self.Proessing(self.traindata)
        # self.SaveData()
        self.Train()

    def Train(self):
        print "train model"
        # resultfile = r"H:\Kaggle\data\Titanic\result.csv"
        # self.SaveData(resultfile, data=self.traindata)
        # self.estimater = RandomForestClassifier(n_estimators=2000, n_jobs=3, bootstrap=True)
        # self.estimater=mixture.GMM(n_components=2)
        # self.estimater = MultinomialNB()
        self.estimater=DecisionTreeClassifier()
        self.estimater.fit(self.traindata, self.labldata)

    def GetResult(self, testfile, filename):
        print "load test data"
        data = pd.read_csv(testfile)
        indexs = data["PassengerId"]
        self.testdata = self.__SetData(data)
        self.Proessing(self.testdata)
        resultfile = r"H:\Kaggle\data\Titanic\result.csv"
        # self.SaveData(resultfile,data=self.testdata,)
        # self.testdata.dropna(how="all")

        result = self.estimater.predict(self.testdata)
        result = result.T
        index = result.shape[0]
        resutldata = pd.DataFrame(result, index=np.arange(1, index + 1), columns=['Survived'])
        resutldata.index = indexs
        resutldata.to_csv(filename)

    def SaveModel(self, filename):
        joblib.dump(self.estimater, filename, compress=3)

    def LoadModel(self, filename):
        self.estimater = joblib.load(filename)


if __name__ == '__main__':
    trainfile = r"H:\Kaggle\data\Titanic\train.csv"
    testfile = r"H:\Kaggle\data\Titanic\test.csv"
    resultfile = r"H:\Kaggle\data\Titanic\result.csv"
    model = r"H:\Kaggle\data\Titanic\model.pkl"
    deal = Titanic()
    deal.LoadTrainData(trainfile)
    deal.Run()
    # deal.SaveModel(model)
    # deal.LoadTestData(testfile)
    deal.GetResult(testfile, resultfile)
    # deal.SaveData(savafile, deal.traindata)
