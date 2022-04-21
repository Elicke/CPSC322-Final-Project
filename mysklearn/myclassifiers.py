import operator
import math
from mysklearn import myutils

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        
        for test_instance in X_test:
            curr_row_indexes_dists = []
            for i, train_instance in enumerate(self.X_train):
                dist = myutils.compute_euclidean_distance(train_instance, test_instance)
                dist = round(dist, 3)
                curr_row_indexes_dists.append([i, dist])
            curr_row_indexes_dists.sort(key=operator.itemgetter(-1))
            curr_top_k = curr_row_indexes_dists[:self.n_neighbors]
            curr_top_k_dists = []
            curr_top_k_neighbor_indices = []
            for neighbor in curr_top_k:
                curr_top_k_dists.append(neighbor[-1])
                curr_top_k_neighbor_indices.append(neighbor[0])
            distances.append(curr_top_k_dists)
            neighbor_indices.append(curr_top_k_neighbor_indices)
        
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        distances, neighbor_indices = self.kneighbors(X_test)

        for i, test_instance in enumerate(X_test):
            curr_k_nearest_neighbors_indices = neighbor_indices[i]
            curr_k_nearest_neighbors_y_labels = []
            for index in curr_k_nearest_neighbors_indices:
                curr_k_nearest_neighbors_y_labels.append(self.y_train[index])
            y_label_values, y_label_counts = myutils.get_frequencies(curr_k_nearest_neighbors_y_labels)
            highest_count_index = y_label_counts.index(max(y_label_counts))
            y_predicted.append(y_label_values[highest_count_index])
        
        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.
    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.
        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        y_train_values, y_train_counts = myutils.get_frequencies(y_train)
        highest_count_index = y_train_counts.index(max(y_train_counts))
        self.most_common_label = y_train_values[highest_count_index]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for test_instance in X_test:
            y_predicted.append(self.most_common_label)
        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.
    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.
    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None
        
        self.possible_class_values = None
        self.possible_values_per_attribute = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """

        class_values, class_counts = myutils.get_frequencies(y_train)
        self.possible_class_values = class_values.copy()
        self.priors = [round((class_count / len(y_train)), 2) for class_count in class_counts]

        # determine size of posteriors table
        num_diff_attr_values = 0
        for a, _ in enumerate(X_train[0]):
            attr_col = myutils.get_column(X_train, known_col_index=a)
            attr_values, _ = myutils.get_frequencies(attr_col)
            num_diff_attr_values += len(attr_values)
        self.posteriors = [[None] * len(class_values) for _ in range(num_diff_attr_values)]

        self.possible_values_per_attribute = []

        # create the posteriors table
        posteriors_row_index = 0
        for i, _ in enumerate(X_train[0]): # for each type of attribute
            attr_col = myutils.get_column(X_train, known_col_index=i)
            attr_values, _ = myutils.get_frequencies(attr_col)
            self.possible_values_per_attribute.append(attr_values)
            for attr_value in attr_values: # for each different possible value of this attribute type
                posteriors_col_index = 0
                for j, class_value in enumerate(class_values): # for each different possible class value
                    attr_class_value_pairs_found = 0
                    for q in range(len(attr_col)):
                        if attr_col[q] == attr_value and y_train[q] == class_value: # count up instances of this attribute value and class value combination
                            attr_class_value_pairs_found += 1
                    self.posteriors[posteriors_row_index][posteriors_col_index] = round(attr_class_value_pairs_found / class_counts[j], 2)
                    posteriors_col_index += 1
                posteriors_row_index += 1

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        y_predicted = []

        for ti, test_instance in enumerate(X_test):
            max_calc = None
            max_calc_class_index = None
            for pcv, possible_class_value in enumerate(self.possible_class_values):
                running_calc = self.priors[pcv]
                for tiv, test_instance_value in enumerate(test_instance):
                    posteriors_row_index = 0
                    for i in range(tiv):
                        posteriors_row_index += len(self.possible_values_per_attribute[i])
                    posteriors_row_index += self.possible_values_per_attribute[tiv].index(test_instance_value)
                    posteriors_col_index = pcv
                    running_calc *= self.posteriors[posteriors_row_index][posteriors_col_index]
                if max_calc == None or running_calc > max_calc:
                    max_calc = running_calc
                    max_calc_class_index = pcv
            y_predicted.append(self.possible_class_values[max_calc_class_index])

        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

        self.header = None
        self.attribute_domains = None

    def select_attribute(self, instances, attributes):
        # TODO: use entropy to calculate and choose the
        # attribute with the smallest Enew
        # for now, we use random attribute selection
        smallest_Enew = None
        smallest_Enew_att = None
        for att in attributes:
            Enew = 0.0
            partitions = self.partition_instances(instances, att)
            for att_value, att_partition in partitions.items():
                Eatt_value = 0.0
                class_col = myutils.get_column(att_partition, known_col_index=-1)
                class_labels, class_label_counts = myutils.get_frequencies(class_col)
                for i, class_label in enumerate(class_labels):
                    if class_label_counts[i] != 0:
                        Eatt_value -= (class_label_counts[i] / len(att_partition)) * \
                            math.log((class_label_counts[i] / len(att_partition)), 2)
                Enew += (len(att_partition) / len(instances)) * Eatt_value
            if smallest_Enew == None or Enew < smallest_Enew:
                smallest_Enew = Enew
                smallest_Enew_att = att
        
        return smallest_Enew_att

        # rand_index = np.random.randint(0, len(attributes))
        # return attributes[rand_index]

    def partition_instances(self, instances, split_attribute):
        # this is a group by attribute domain
        # let's use a dictionary
        partitions = {} # key (attribute value): value (subtable)
        att_index = self.header.index(split_attribute) # e.g. level -> 0
        att_domain = self.attribute_domains["att" + str(att_index)] # e.g. ["Junior", "Mid", "Senior"]
        for att_value in att_domain:
            partitions[att_value] = []
            # task: finish
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions

    def all_same_class(self, att_partition):
        first_class = att_partition[0][-1]
        for instance in att_partition:
            if instance[-1] != first_class:
                return False
        return True

    def handle_clash(self, att_partition):
        classes_in_att_partition = myutils.get_column(att_partition, known_col_index=-1)
        class_values, class_counts = myutils.get_frequencies(classes_in_att_partition)
        majority_class = class_values[class_counts.index(max(class_counts))]
        return majority_class

    def tdidt(self, current_instances, available_attributes):
        # basic approach (uses recursion!!):
        # print("available attributes:", available_attributes)

        # select an attribute to split on
        attribute = self.select_attribute(current_instances, available_attributes)
        # print("splitting on:", attribute)
        available_attributes.remove(attribute) # can't split on this again in
        # this subtree
        tree = ["Attribute", attribute] # start to build the tree!!

        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, attribute)
        # print("partitions:", partitions)
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value, att_partition in partitions.items():
            # print("current att value:", att_value, len(att_partition))
            value_subtree = ["Value", att_value]
            #    CASE 1: all class labels of the partition are the same => make a leaf node
            if len(att_partition) > 0 and self.all_same_class(att_partition):
                # print("CASE 1 all same class")
                # TODO: make a leaf node
                leaf = ["Leaf", att_partition[0][-1], len(att_partition), len(current_instances)]
                value_subtree.append(leaf)
            #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                # print("CASE 2 no more attributes")
                # TODO: we have a mix of class labels, handle clash w/
                # majority vote leaf node
                majority_class_label = self.handle_clash(att_partition)
                leaf = ["Leaf", majority_class_label, len(att_partition), len(current_instances)]
                value_subtree.append(leaf)
            #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                # print("CASE 3 empty partition")
                # TODO: "backtrack" and replace this attribute node
                # with a majority vote leaf node
                majority_class_label = self.handle_clash(current_instances)
                tree = ["Leaf", majority_class_label, len(current_instances), None]
                break
            else: # none of the previous conditions were true... recurse!
                subtree = self.tdidt(att_partition, available_attributes.copy())
                # note the copy
                if subtree[0] == "Leaf":
                    subtree[3] = len(current_instances)
                # TODO: append subtree to value_subtree and tree
                # appropriately
                value_subtree.append(subtree)
            tree.append(value_subtree)
        
        ##### case 3 example => no 6/15???
        return tree

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        self.header = []
        self.attribute_domains = {}
        # TODO: programmatically create a header (e.g. ["att0", "att1",
        # ...] and create an attribute domains dictionary)
        for i in range(len(X_train[0])):
            self.header.append("att" + str(i))
        for j, att_label in enumerate(self.header):
            att_column = myutils.get_column(X_train, known_col_index=j)
            att_domain, _ = myutils.get_frequencies(att_column)
            self.attribute_domains[att_label] = att_domain
        
        # next, I advise stitching X_train and y_train together
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # now, making a copy of the header because tdidt()
        # is going to modify the list
        available_attributes = self.header.copy()
        # recall: python is pass by object reference
        self.tree = self.tdidt(train, available_attributes)
        # print("tree:", self.tree)
        # note: the unit test will assert tree == interview_tree_solution
        # (mind the attribute value order)

    def tdidt_predict(self, curr_tree, test_instance):
        # recursively traverse the tree
        # we need to know where we are in the tree...
        # are we at a leaf node (base case) or
        # attribute node
        info_type = curr_tree[0]
        if info_type == "Leaf":
            return curr_tree[1]
        # we need to match the attribute's value in the
        # instance with the appropriate value list
        # in the tree
        # a for loop that traverses through
        # each value list
        # recurse on match with instance's value
        att_index = self.header.index(curr_tree[1])
        for i in range(2, len(curr_tree)):
            value_list = curr_tree[i]
            if value_list[1] == test_instance[att_index]:
                # we have a match, recurse
                return self.tdidt_predict(value_list[2], test_instance)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for test_instance in X_test:
            y_predicted.append(self.tdidt_predict(self.tree, test_instance))

        return y_predicted

    def extract_decision_rules(self, curr_tree, curr_rule_string, attribute_names, class_name):

        if curr_tree[0] == "Leaf":
            curr_rule_string += "THEN " + class_name + " = " + curr_tree[1]
            print(curr_rule_string)
        else:
            if curr_rule_string == "":
                curr_rule_string += "IF "
            else:
                curr_rule_string += "AND "
            if attribute_names == None:
                att_name = curr_tree[1]
            else:
                att_name = attribute_names[self.header.index(curr_tree[1])]
            curr_rule_string += att_name + " == "
            saved_string = str(curr_rule_string)
            for i in range(2, len(curr_tree)):
                curr_rule_string = str(saved_string)
                curr_value_subtree = curr_tree[i]
                curr_rule_string += curr_value_subtree[1] + " "
                self.extract_decision_rules(curr_value_subtree[2], curr_rule_string, attribute_names, class_name)

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        self.extract_decision_rules(self.tree, "", attribute_names, class_name)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this