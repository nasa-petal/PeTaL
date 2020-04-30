from bases import DimReducer
from sklearn.decomposition import PCA as PCASklearn
from sklearn.decomposition import FastICA
import pickle

class PCA(DimReducer):
    recuder_hdf = pickle.load( open("parameters.pickle", "rb" ) )
    recuder_hdf = recuder_hdf.loc[recuder_hdf['model'] == 'PCA'].drop("model", axis=1)

    def __init__(self, recuder_hdf):
        self._reducer = PCASklearn()
        
        super().__init__(recuder_hdf)


class ICA(DimReducer):
    recuder_hdf = pickle.load( open("parameters.pickle", "rb" ) )
    recuder_hdf = recuder_hdf.loc[recuder_hdf['model'] == 'ICA'].drop("model", axis=1)

    def __init__(self, recuder_hdf):
        self._reducer = FastICA()
        
        super().__init__(recuder_hdf)

