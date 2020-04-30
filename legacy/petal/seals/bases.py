from itertools import zip_longest
from sklearn.preprocessing import LabelEncoder


class Entity():
    hyperparam_df = None

    def __init__(self, hyperparam_df):
        self.hyperparam_df = hyperparam_df
        super().__init__()

    def hdf_to_dict(self):
        '''
        Converts the hyperparameter dataframe to a dictionary

        Returns
        -------
        `dict`
            The dictionary of hyperparameters and current values
        '''

        dictionary = {}
        for param, value in self.hyperparam_df[['param','current_value']].itertuples(index=False):
            dictionary[param] = value
        return dictionary

    def get_param(self, param):
        '''
        Returns current value of hyperparameter `param`

        Parameters
        ----------
        param: `str`
            The name of the parameter whose value is to be returned
        
        Returns
        -------
        `any`
            The current value of the parameter
        '''
        return self.hyperparam_df.loc[self.hyperparam_df['param'] == param, 'current_value'].iloc[0]

class Model(Entity):
    _model = None
    _encoders = []
    _dim_reducer = None
    hyperparam_df = None
    supervised = True #True for supervised, False for unsupervised

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val):
        self._model=val

    @property
    def encoders(self):
        return self._encoders

    @encoders.setter
    def encoders(self, val):
        self._encoders=val

    @property
    def dim_reducer(self):
        return self._dim_reducer

    @dim_reducer.setter
    def dim_reducer(self, val):
        self._dim_reducer=val

    def __init__(self, hyperparam_df, dim_class, dimhyperparam_df):
        if dim_class:
            self._dim_reducer = dim_class(dimhyperparam_df)
        super().__init__(hyperparam_df)

    def update_hyperparams(self):
        '''
        Uses sklearn's set_params to update the models hyperparameters if the model has this function
        '''
        set_params = getattr(self._model, "set_params", None)
        if not callable(set_params):
            raise("Model has no function 'set_params', so update_hyperparams cannot be used.")
        self._model = set_params(**self.hdf_to_dict())


    def fit(self, data, fit_funct):
        '''
        Fits the model to the data. If there is a dimesionality reducer, that will be fit as well

        Parameters
        ----------
        data: `list` [ `list` [`any`]]
            A list of examples
        fit_funct: `function`
            The model's fit function
        '''
        if type(data) == tuple:
            X, y = data
        else:
            X = data

        if self.dim_reducer:
            # if there is a dimensionality reducer on the model, fit that model then transform the data
            self.dim_reducer.fit(X)

            X = self.dim_reducer.transform(X)

        if self.supervised:
            fit_funct(X,y)
        else:
            fit_funct(X)
        
    def classify_data(self, data, pred_funct):
        '''
        Classifies the given examples using the current model (and reduces the data if necessary)

        Parameters
        ----------
        data: `list` [ `list` [`any`]]
            A list of examples
        pred_funct: `function`
            The model's prediction funciton
        '''
        self.check_model()
        if self._dim_reducer:
            data = self._dim_reducer.transform(data)
        return pred_funct(data)

    def check_model(self):
        if not self._model:
            raise Exception('Model does not exist')

        try:
            if not self.hyperparam_df:
                raise Exception('No hyperparameter dictionary')
        except ValueError:
            # truth value of a DataFrame is ambiguous
            pass

    def get_param(self, param):
        '''
        Returns current value of hyperparameter `param`

        Parameters
        ----------
        param: `str`
            The name of the parameter whose value is to be returned
        
        Returns
        -------
        `any`
            The current value of the parameter
        '''
        return self.hyperparam_df.loc[self.hyperparam_df['param'] == param, 'current_value'].iloc[0]

    def encode_categorical(self, df):
        '''
        If the model has already trained encoders, encode `df`. If not, train them on `df` then encode it.
        
        Parameters
        ----------
        df: `DataFrame`
            The dataframe of values to be encoded

        Returns
        -------
        `DataFrame`
            The encoded dataframe
        '''

        if self.encoders:
            # model already has label encoders
            encoder_list = self.encoders
            for enc, col in zip_longest(encoder_list, df.columns):
                if enc and col:
                    df[col] =  enc.transform(df[col])
        else:
            encoder_list = []
            for col in df.columns:
                enc = LabelEncoder()
                if df[col].dtype == "object":
                    # if feature has a type that needs to be encoded, train an encoder
                    enc.fit(df[col])
                    encoder_list.append(enc)
                    df[col] = enc.transform(df[col]) # transform the feature with the trained encoder
                else:
                    encoder_list.append(None)
            self.encoders = encoder_list

        return df

    def decode_categorical(self, df):
        '''
        Decodes data that have encoders on them
        
        Parameters
        ----------
        df: `DataFrame`
            The dataframe of values to be decoded

        Returns
        -------
        `DataFrame`
            The decoded dataframe
        '''

        if not self.encoders:
            raise Exception('Attempting decode before encoders are trained')

        encoder_list = self.encoders
        
        for enc, col in zip_longest(encoder_list, df.columns):
            if enc and col:
                df[col] =  enc.inverse_transform(df[col])
        return df


class DimReducer(Entity):
    _reducer = None

    @property
    def reducer(self):
        return self._reducer

    @reducer.setter
    def reducer(self, val):
        self._reducer=val

    def __init__(self, reducer_hdf):
        super().__init__(reducer_hdf)
        self._reducer = self._reducer.set_params(**self.hdf_to_dict())
        

    def fit(self, data):
        self.reducer.fit(data)
    
    def transform(self, data):
        return self.reducer.transform(data)