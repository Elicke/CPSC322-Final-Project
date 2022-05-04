import pickle

from mysklearn.myclassifiers import MyNaiveBayesClassifier
from mysklearn.mypytable import MyPyTable
import mysklearn.myutils as myutils

pytable_10s = MyPyTable().load_from_file("input_data/dataset-of-10s.csv")
pytable_00s = MyPyTable().load_from_file("input_data/dataset-of-00s.csv")

_, target_subtables_10s = myutils.group_by(pytable_10s.data, pytable_10s.column_names, "target")
_, target_subtables_00s = myutils.group_by(pytable_00s.data, pytable_00s.column_names, "target")

reduced_size_table = []
for i in range(500):
    reduced_size_table.append(target_subtables_10s[0][i])
    reduced_size_table.append(target_subtables_10s[1][i])
    reduced_size_table.append(target_subtables_00s[0][i])
    reduced_size_table.append(target_subtables_00s[1][i])

tempo_col = myutils.get_column(reduced_size_table, known_col_index=13) # Needed for normalization
X_data = ([[track[3], track[4], track[12], myutils.normalize_tempo_value(track[13], min(tempo_col), max(tempo_col))]
    for track in reduced_size_table])
discretized_X_data = ([[myutils.discretize_value(instance[0]), myutils.discretize_value(instance[1]),
    myutils.discretize_value(instance[2]), myutils.discretize_value(instance[3])] for instance in X_data])
y_data = [track[-1] for track in reduced_size_table]

nb_classifier = MyNaiveBayesClassifier()
nb_classifier.fit(discretized_X_data, y_data)
packaged_obj = [nb_classifier, min(tempo_col), max(tempo_col)]
outfile = open("nb.p", "wb")
pickle.dump(packaged_obj, outfile)
outfile.close()