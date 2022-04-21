from mysklearn import myutils

import math
import numpy as np
from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!
    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """

    if random_state != None:
        np.random.seed(random_state)
    
    X_copy = X.copy()
    y_copy = y.copy()
    if shuffle:
        myutils.randomize_in_place(X_copy, y_copy)
    
    if isinstance(test_size, int):
        num_test_instances = test_size
    else:
        num_test_instances = test_size * len(X_copy)
        num_test_instances = math.ceil(num_test_instances)
    
    X_train = X_copy[:(len(X) - num_test_instances)]
    X_test = X_copy[(len(X) - num_test_instances):]
    y_train = y_copy[:(len(X) - num_test_instances)]
    y_test = y_copy[(len(X) - num_test_instances):]

    return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold
    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    if random_state != None:
        np.random.seed(random_state)
    
    X_indexes = list(range(len(X)))
    if shuffle:
        myutils.randomize_in_place(X_indexes)
    
    num_first_folds = len(X_indexes) % n_splits
    first_folds_size = len(X_indexes) // n_splits + 1
    other_folds_size = len(X_indexes) // n_splits

    X_train_folds = []
    X_test_folds = []
    curr_dataset_index = 0
    for i in range(n_splits):
        curr_X_test_fold = []
        if i + 1 <= num_first_folds:
            for _ in range(first_folds_size):
                curr_X_test_fold.append(X_indexes[curr_dataset_index])
                curr_dataset_index += 1
        else:
            for _ in range(other_folds_size):
                curr_X_test_fold.append(X_indexes[curr_dataset_index])
                curr_dataset_index += 1
        X_test_folds.append(curr_X_test_fold)
        curr_X_train_fold = []
        for j in range(len(X_indexes)):
            if X_indexes[j] not in curr_X_test_fold:
                curr_X_train_fold.append(X_indexes[j])
        X_train_folds.append(curr_X_train_fold)

    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.
    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.
    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """

    if random_state != None:
        np.random.seed(random_state)

    X_indexes = list(ind for ind in range(len(X)))
    twoD_X_indexes = list([i] for i in range(len(X)))

    y_copy = y.copy()
    if shuffle:
        myutils.randomize_in_place(twoD_X_indexes, y_copy)

    sample_indexes_with_class_labels = twoD_X_indexes.copy()
    for j in range(len(sample_indexes_with_class_labels)):
        sample_indexes_with_class_labels[j].append(y_copy[j])
    header_for_group_by = ["", "y"]

    _, group_subtables = myutils.group_by(sample_indexes_with_class_labels, header_for_group_by, "y")

    X_train_folds = list([] for _ in range(n_splits))
    X_test_folds = list([] for _ in range(n_splits))

    for group_subtable in group_subtables:
        for s, sample in enumerate(group_subtable):
            X_test_folds[s % n_splits].append(sample[0])

    for t in range(len(X_train_folds)):
        for q in range(len(X_indexes)):
            if X_indexes[q] not in X_test_folds[t]:
                X_train_folds[t].append(X_indexes[q])

    return X_train_folds, X_test_folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.
    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """

    if random_state != None:
        np.random.seed(random_state)
    
    if n_samples == None:
        n_samples = len(X)
    
    X_sample = []
    if y != None:
        y_sample = []
    else:
        y_sample = None
    for _ in range(n_samples):
        rand_index = np.random.randint(len(X))
        X_sample.append(X[rand_index])
        if y != None:
            y_sample.append(y[rand_index])
    
    X_out_of_bag = []
    if y != None:
        y_out_of_bag = []
    else:
        y_out_of_bag = None
    for i, instance in enumerate(X):
        if instance not in X_sample:
            X_out_of_bag.append(instance)
            if y != None:
                y_out_of_bag.append(y[i])

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class
    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """

    matrix = list([0 for _ in range(len(labels))] for _ in range(len(labels)))

    for t in range(len(y_true)):
        i = labels.index(y_true[t])
        j = labels.index(y_pred[t])
        matrix[i][j] += 1

    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).
    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """

    correct_count = 0

    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct_count += 1
    
    if normalize:
        return correct_count / len(y_pred)
    else:
        return correct_count

def do_random_subsampling(k, test_size, X, y, classifier, normalize_X=False):
    accuracies_combined = 0.0
    for iteration in range(k):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)
        if normalize_X:
            normalized_X_train = [[None for _ in range(len(X_train[0]))] for _ in range(len(X_train))]
            for column_index in range(len(X_train[0])):
                curr_column = myutils.get_column(X_train, known_col_index=column_index)
                curr_column = myutils.normalize_training_data(curr_column)
                for i in range(len(X_train)):
                    normalized_X_train[i][column_index] = curr_column[i]
            normalized_X_test = [[None for _ in range(len(X_test[0]))] for _ in range(len(X_test))]
            for column_ind in range(len(X_test[0])):
                curr_column = myutils.get_column(X_test, known_col_index=column_ind)
                curr_column_training = myutils.get_column(X_train, known_col_index=column_ind)
                curr_column = myutils.normalize_test_data(curr_column, curr_column_training)
                for j in range(len(X_test)):
                    normalized_X_test[j][column_ind] = curr_column[j]
        if normalize_X:
            classifier.fit(normalized_X_train, y_train)
            y_predicted = classifier.predict(normalized_X_test)
        else:
            classifier.fit(X_train, y_train)
            y_predicted = classifier.predict(X_test)
        accuracies_combined += accuracy_score(y_test, y_predicted)
    accuracy = accuracies_combined / k
    error_rate = 1.0 - accuracy
    return accuracy, error_rate

def do_cross_validation(k, X, y, classifier, normalize_X=False, stratified=False, pos_label=None):
    if stratified:
        X_train_folds, X_test_folds = stratified_kfold_cross_validation(X, y, k, shuffle=True)
    else:
        X_train_folds, X_test_folds = kfold_cross_validation(X, k, shuffle=True)
    num_correct_classifications = 0
    all_y_true = []
    all_y_pred = []
    for fold_iteration in range(k):
        curr_X_train_indexes = X_train_folds[fold_iteration]
        curr_X_test_indexes = X_test_folds[fold_iteration]
        X_train = list(X[X_train_index] for X_train_index in curr_X_train_indexes)
        X_test = list(X[X_test_index] for X_test_index in curr_X_test_indexes)
        y_train = list(y[y_train_index] for y_train_index in curr_X_train_indexes)
        y_test = list(y[y_test_index] for y_test_index in curr_X_test_indexes)
        if normalize_X:
            normalized_X_train = [[None for _ in range(len(X_train[0]))] for _ in range(len(X_train))]
            for column_index in range(len(X_train[0])):
                curr_column = myutils.get_column(X_train, known_col_index=column_index)
                curr_column = myutils.normalize_training_data(curr_column)
                for i in range(len(X_train)):
                    normalized_X_train[i][column_index] = curr_column[i]
            normalized_X_test = [[None for _ in range(len(X_test[0]))] for _ in range(len(X_test))]
            for column_ind in range(len(X_test[0])):
                curr_column = myutils.get_column(X_test, known_col_index=column_ind)
                curr_column_training = myutils.get_column(X_train, known_col_index=column_ind)
                curr_column = myutils.normalize_test_data(curr_column, curr_column_training)
                for j in range(len(X_test)):
                    normalized_X_test[j][column_ind] = curr_column[j]
        if normalize_X:
            classifier.fit(normalized_X_train, y_train)
            y_predicted = classifier.predict(normalized_X_test)
        else:
            classifier.fit(X_train, y_train)
            y_predicted = classifier.predict(X_test)
        num_correct_classifications += accuracy_score(y_test, y_predicted, normalize=False)
        for y_true_val in y_test:
            all_y_true.append(y_true_val)
        for y_pred_val in y_predicted:
            all_y_pred.append(y_pred_val)
    accuracy = num_correct_classifications / len(X)
    error_rate = 1.0 - accuracy
    precision = binary_precision_score(all_y_true, all_y_pred, pos_label=pos_label)
    recall = binary_recall_score(all_y_true, all_y_pred, pos_label=pos_label)
    f1 = binary_f1_score(all_y_true, all_y_pred, pos_label=pos_label)
    return accuracy, error_rate, precision, recall, f1, all_y_true, all_y_pred

def do_bootstrap_method(k, X, y, classifier, normalize_X=False):
    accuracies_combined = 0.0
    for iteration in range(k):
        X_train, X_test, y_train, y_test = bootstrap_sample(X, y)
        if normalize_X:
            normalized_X_train = [[None for _ in range(len(X_train[0]))] for _ in range(len(X_train))]
            for column_index in range(len(X_train[0])):
                curr_column = myutils.get_column(X_train, known_col_index=column_index)
                curr_column = myutils.normalize_training_data(curr_column)
                for i in range(len(X_train)):
                    normalized_X_train[i][column_index] = curr_column[i]
            normalized_X_test = [[None for _ in range(len(X_test[0]))] for _ in range(len(X_test))]
            for column_ind in range(len(X_test[0])):
                curr_column = myutils.get_column(X_test, known_col_index=column_ind)
                curr_column_training = myutils.get_column(X_train, known_col_index=column_ind)
                curr_column = myutils.normalize_test_data(curr_column, curr_column_training)
                for j in range(len(X_test)):
                    normalized_X_test[j][column_ind] = curr_column[j]
        if normalize_X:
            classifier.fit(normalized_X_train, y_train)
            y_predicted = classifier.predict(normalized_X_test)
        else:
            classifier.fit(X_train, y_train)
            y_predicted = classifier.predict(X_test)
        accuracies_combined += accuracy_score(y_test, y_predicted)
    accuracy = accuracies_combined / k # could not figure out how to calculate the *weighted* average accuracy
    error_rate = 1.0 - accuracy
    return accuracy, error_rate

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        precision(float): Precision of the positive class
    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """

    if labels == None:
        labels, _ = myutils.get_frequencies(y_true)
    
    if pos_label == None:
        pos_label = labels[0]
    
    tp = 0
    fp = 0
    for i in range(len(y_pred)):
        if y_pred[i] == pos_label:
            if y_true[i] == pos_label:
                tp += 1
            else:
                fp += 1
    
    if (tp + fp) == 0:
        return 0.0

    precision = tp / (tp + fp)

    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        recall(float): Recall of the positive class
    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """

    if labels == None:
        labels, _ = myutils.get_frequencies(y_true)
    
    if pos_label == None:
        pos_label = labels[0]
    
    tp = 0
    fn = 0
    for i in range(len(y_pred)):
        if y_pred[i] == pos_label and y_true[i] == pos_label:
            tp += 1
        elif y_pred[i] != pos_label and y_true[i] == pos_label:
            fn += 1
    
    if (tp + fn) == 0:
        return 0.0
    
    recall = tp / (tp + fn)

    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        f1(float): F1 score of the positive class
    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """

    if labels == None:
        labels, _ = myutils.get_frequencies(y_true)
    
    if pos_label == None:
        pos_label = labels[0]
    
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    if (precision + recall) == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1