import numpy as np

def get_column(table, header=None, col_name=None, known_col_index=None):
    if known_col_index != None:
        col_index = known_col_index
    else:
        col_index = header.index(col_name)
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col

def get_frequencies(list_of_values):
    list_of_values_copy = list_of_values.copy()
    list_of_values_copy.sort() 
    # parallel lists
    values = []
    counts = []
    for value in list_of_values_copy:
        if value in values: # seen it before
            counts[-1] += 1 # okay because sorted
        else: # haven't seen it before
            values.append(value)
            counts.append(1)

    return values, counts

def compute_euclidean_distance(v1, v2):
    # return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

    running_sum = 0
    for i in range(len(v1)):
        if isinstance(v1[i], str):
            if v1[i] == v2[i]:
                current_calc = 0
            else:
                current_calc = 1
        else:
            current_calc = (v1[i] - v2[i]) ** 2
        running_sum += current_calc
    return np.sqrt(running_sum)

def normalize_training_data(training_values):
    max_value = max(training_values)
    min_value = min(training_values)
    value_range = max_value - min_value

    normalized_values = []
    for value in training_values:
        normalized_values.append((value - min_value) / value_range)
    
    return normalized_values

def normalize_test_data(test_values, training_values):
    max_value = max(training_values)
    min_value = min(training_values)
    value_range = max_value - min_value

    normalized_values = []
    for value in test_values:
        normalized_values.append((value - min_value) / value_range)
    
    return normalized_values

def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap values with 
        rand_index = np.random.randint(0, len(alist)) # [0, len(alist))
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] =\
                parallel_list[rand_index], parallel_list[i]

def group_by(table, header, groupby_col_name):
    groupby_col_index = header.index(groupby_col_name) # use this later
    groupby_col = get_column(table, header, groupby_col_name)
    group_names = sorted(list(set(groupby_col))) # e.g. [75, 76, 77]
    group_subtables = [[] for _ in group_names] # e.g. [[], [], []]
    
    for row in table:
        groupby_val = row[groupby_col_index] # e.g. this row's modelyear
        # which subtable does this row belong?
        groupby_val_subtable_index = group_names.index(groupby_val)
        group_subtables[groupby_val_subtable_index].append(row.copy()) # make a copy
    
    return group_names, group_subtables

def compute_equal_width_cutoffs(values, num_bins):
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins
    cutoffs = list(np.arange(min(values), max(values), bin_width))
    cutoffs.append(max(values))
    cutoffs = [round(cutoff, 1) for cutoff in cutoffs]
    return cutoffs

def compute_bin_frequencies(values, cutoffs):
    freqs = [0 for _ in range(len(cutoffs) - 1)]

    for value in values:
        if value == max(values):
            freqs[-1] += 1 # increment the last bin's freq
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= value < cutoffs[i + 1]:
                    freqs[i] += 1 
    return freqs

def compute_slope_intercept(x, y):
    meanx = np.mean(x)
    meany = np.mean(y)

    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    den = sum([(x[i] - meanx) ** 2 for i in range(len(x))])
    m = num / den 
    # y = mx + b => b = y - mx
    b = meany - m * meanx
    return m, b

def compute_correlation_coefficient(x, y):
    meanx = np.mean(x)
    meany = np.mean(y)

    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    den = np.sqrt(sum([(x[i] - meanx) ** 2 for i in range(len(x))]) * sum([(y[i] - meany) ** 2 for i in range(len(y))]))
    r = num / den
    return r

def compute_covariance(x, y):
    meanx = np.mean(x)
    meany = np.mean(y)

    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    den = len(x)
    cov = num / den
    return cov

###########################################################################################################################

def normalize_tempo_value(tempo_val, min_tempo, max_tempo):
    return (tempo_val - min_tempo) / (max_tempo - min_tempo)

def discretize_value(val):
    if val >= 0.8:
        return 5
    elif val >= 0.6:
        return 4
    elif val >= 0.4:
        return 3
    elif val >= 0.2:
        return 2
    else:
        return 1