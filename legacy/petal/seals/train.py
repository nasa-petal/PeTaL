import sys, pickle, pandas, inspect, itertools
import cv
import models, dim_reducers
from models import *
from dim_reducers import *
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from hyperparams import get_hyperparam_selections

NUM_REQ_ARGS = 3
AVAILABLE_MODELS = set([m[0] for m in inspect.getmembers(models, inspect.isclass) if m[1].__module__ == 'models'])
AVAILABLE_DIM_REDUCERS = set([m[0] for m in inspect.getmembers(dim_reducers, inspect.isclass) if m[1].__module__ == 'dim_reducers'])
REQ_MESSAGE = "Inappropriate number of arguments. Required arguments are:\n model name\n path to training data\n desired path for output model\n number of cross-validation folds (only required for the supervised case. 0 for no folds)"

def main():
    if len(sys.argv) - 1 < NUM_REQ_ARGS:
        print(REQ_MESSAGE)
        return

    model_name = str(sys.argv[1])
    data_path = str(sys.argv[2])
    pickle_path = str(sys.argv[3])
    num_folds = None

    if model_name not in AVAILABLE_MODELS:
        print("Unrecognized model: ", model_name, "\nAvailable models are", ", ".join(AVAILABLE_MODELS))
        return

    model_class = getattr(sys.modules[__name__], model_name)
    labels = model_class.supervised

    if labels:
        if len(sys.argv) - 1 < NUM_REQ_ARGS + 1:
            print(REQ_MESSAGE)
            return
        num_folds = int(sys.argv[NUM_REQ_ARGS + 1])

    hyperparam_df = model_class.hyperparam_df
    hyperparam_df = get_hyperparam_selections(hyperparam_df)

    data = shuffle(pandas.read_csv(data_path))

    while True:
        reducer = input("Select dimensionality reducer from the list: " + ", ".join(AVAILABLE_DIM_REDUCERS) + " or press enter to skip: ")
        if reducer and str(reducer) in AVAILABLE_DIM_REDUCERS:
            reducer_class = getattr(sys.modules[__name__], reducer)
            reducer_hdf = get_hyperparam_selections(reducer_class.recuder_hdf)

            model=model_class(hyperparam_df, reducer_class, reducer_hdf)
            break
        elif not reducer:
            # they don't want to set a reducer
            model=model_class(hyperparam_df, None, None)
            break


    data = model.encode_categorical(data)


    if labels:
        print("Using column '"+ str(data.columns[-1]) + "' as labels.\n")
        data_y = data.iloc[:,-1].values.tolist()
        data.drop(data.columns[[-1]], axis=1, inplace=True)
        data_X = data.values.tolist()

        if num_folds > 0:
            cv.run_cv(data_X, num_folds, data_y, len(set(data_y)), model)
        
        model.fit((data_X, data_y)) # the final model to be saved

    else:
        if data.columns[-1].lower() == 'class':
            data.drop(data.columns[[-1]], axis=1, inplace=True)
        data_X = data.values.tolist()

        model.fit(data_X)
        
    pickle.dump(model, open(pickle_path, "wb" ))

if __name__ == "__main__":
    main()    
