from sklearn.metrics import confusion_matrix
import numpy as np


def get_folds(data, numfolds, labels, numclasses):
    '''
    Creates crossvalidation folds

    Parameters
    ----------
    data : `list` [`list` [`any`]]
        contains the data to be folded

    numfolds : `int`
        the number of folds

    labels : `list` [`any`]
        the class labels corresponding to the data
    
    numclasses : `int`
        the number of classes the data fall into

    Returns
    -------
    folded_data : `list` [`list` [`list` [`any`]]]
        numfolds lists of data

    folded_labels : `list` [`list` [[`any`]]
        numfolds lists of labels correspoinding to the folded_data
    '''

    assert(len(data) == len(labels))
    folded_data = [[] for _ in range(numfolds)]
    folded_labels = [[] for _ in range(numfolds)]
    if numfolds <= numclasses:
        # the first class will start on the first fold, the second on the second, etc.
        offset = 1
    else:
        # when there are more folds than classes, we want to spread out the starting fold for each class more
        # this will make the number of data per class more even
        offset = numfolds // numclasses
    next_fold = []
    for y in range(numclasses):
        next_fold.append((y*offset)%numfolds)
    for i in range(len(data)):
        y = labels[i]
        folded_data[next_fold[y]].append(data[i])
        folded_labels[next_fold[y]].append(y)
        next_fold[y] = (next_fold[y] + 1) % numfolds
    
    return folded_data, folded_labels

def run_cv(data, numfolds, labels, numclasses, model):
    '''
    Runs k-fold stratified cross validation and prints accuracy, precision, and recall

    Parameters
    ----------
    data : `list` [`list` [`any`]]
        contains the data the cv will be run on

    numfolds : `int`
        the number of folds

    labels : `list` [`any`]
        the class labels corresponding to the data
    
    numclasses : `int`
        the number of classes the data fall into

    model : `any`
        the model that will take the data and then be tested
    '''
    folded_data, folded_labels = get_folds(data, numfolds, labels, numclasses)
    results_per_fold = []

    for i in range(numfolds):
        validation_data = folded_data[i]
        validation_labels = folded_labels[i]
        training_data = [example for fold in folded_data[:i] + folded_data[i+1:] for example in fold]
        training_labels = [label for fold in folded_labels[:i] + folded_labels[i+1:] for label in fold]

        model.fit((training_data, training_labels))

        estimated_labels = model.classify_data(validation_data).tolist()

        results_per_fold.append(confusion_matrix(validation_labels, estimated_labels))

    accuracies = []
    precisions = []
    recalls = []
    for mat in results_per_fold:
        no_diag = mat.copy()
        np.fill_diagonal(no_diag, 0)

        total = np.sum(mat, axis = 0)
        tp = np.diag(mat)
        fp = np.sum(no_diag, axis = 0)
        fn = np.sum(no_diag, axis = 1)

        accuracies.append([tp[i]/total[i] if total[i] != 0 else np.nan for i in range(numclasses)])
        recalls.append([tp[i]/(tp[i]+fn[i]) if tp[i]+fn[i] != 0 else np.nan for i in range(numclasses)])
        precisions.append([tp[i]/(tp[i]+fp[i]) if tp[i]+fp[i] != 0 else np.nan for i in range(numclasses)])

    print("Accuracy: ", np.nanmean(accuracies), np.nanstd(accuracies))
    print("Precision: ", np.nanmean(precisions), np.nanstd(precisions))
    print("Recall: ", np.nanmean(recalls), np.nanstd(recalls))