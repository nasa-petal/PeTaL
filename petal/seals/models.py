import pandas as pd
import pickle
from bases import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

class ANN(Model):
    hyperparam_df = pickle.load( open("parameters.pickle", "rb" ) ) # load all hyperparameters
    hyperparam_df = hyperparam_df.loc[hyperparam_df['model'] == 'ANN'].drop("model", axis=1) # pull out the hyperparametrs for this model

    def __init__(self, hyperparam_df, dim_class, dim_hdf):
        self._model = MLPClassifier()

        super().__init__(hyperparam_df, dim_class, dim_hdf)
        self.update_hyperparams()

    def fit(self, data):
        super().fit(data, self._model.fit)

    def classify_data(self, data):
        return super().fit(data, self._model.predict)

class RandomForest(Model):
    hyperparam_df = pickle.load( open("parameters.pickle", "rb" ) )
    hyperparam_df = hyperparam_df.loc[hyperparam_df['model'] == 'RandomForest'].drop("model", axis=1)

    def __init__(self, hyperparam_df, dim_class, dim_hdf):
        self._model = RandomForestClassifier()

        super().__init__(hyperparam_df, dim_class, dim_hdf)
        self.update_hyperparams()

    def fit(self, data):
        super().fit(data, self._model.fit)

    def classify_data(self, data):
        return super().classify_data(data, self._model.predict)

class DTree(Model):
    hyperparam_df = pickle.load( open("parameters.pickle", "rb" ) )
    hyperparam_df = hyperparam_df.loc[hyperparam_df['model'] == 'DTree'].drop("model", axis=1)

    def __init__(self, hyperparam_df, dim_class, dim_hdf):
        self._model = DecisionTreeClassifier()

        super().__init__(hyperparam_df, dim_class, dim_hdf)
        self.update_hyperparams()

    def fit(self, data):
        super().fit(data, self._model.fit)

    def classify_data(self, data):
        return super().classify_data(data, self._model.predict)

class KNN(Model):
    hyperparam_df = pickle.load( open("parameters.pickle", "rb" ) )
    hyperparam_df = hyperparam_df.loc[hyperparam_df['model'] == 'KNN'].drop("model", axis=1)

    def __init__(self, hyperparam_df, dim_class, dim_hdf):
        self._model = KNeighborsClassifier()

        super().__init__(hyperparam_df, dim_class, dim_hdf)
        self.update_hyperparams()

    def fit(self, data):
        super().fit(data, self._model.fit)

    def classify_data(self, data):
        return super().classify_data(data, self._model.predict)

class SVM(Model):
    hyperparam_df = pickle.load( open("parameters.pickle", "rb" ) )
    hyperparam_df = hyperparam_df.loc[hyperparam_df['model'] == 'SVM'].drop("model", axis=1)

    def __init__(self, hyperparam_df, dim_class, dim_hdf):
        self._model = SVC()

        super().__init__(hyperparam_df, dim_class, dim_hdf)
        self.update_hyperparams()

    def fit(self, data):
        super().fit(data, self._model.fit)

    def classify_data(self, data):
        return super().classify_data(data, self._model.predict)

class LogReg(Model):
    hyperparam_df = pickle.load( open("parameters.pickle", "rb" ) )
    hyperparam_df = hyperparam_df.loc[hyperparam_df['model'] == 'LogReg'].drop("model", axis=1)

    def __init__(self, hyperparam_df, dim_class, dim_hdf):
        self._model = LogisticRegression()
        
        super().__init__(hyperparam_df, dim_class, dim_hdf)
        self.update_hyperparams()

    def fit(self, data):
        super().fit(data, self._model.fit)

    def classify_data(self, data):
        return super().classify_data(data, self._model.predict)


class kMeans(Model):
    supervised = False
    hyperparam_df = pickle.load( open("parameters.pickle", "rb" ) )
    hyperparam_df = hyperparam_df.loc[hyperparam_df['model'] == 'kMeans'].drop("model", axis=1)

    def __init__(self, hyperparam_df, dim_class, dim_hdf):
        self._model = KMeans()

        super().__init__(hyperparam_df, dim_class, dim_hdf)
        self.update_hyperparams()

    def fit(self, data):
        super().fit(data, self._model.fit)

    def classify_data(self, data):
        return super().classify_data(data, self._model.predict)

