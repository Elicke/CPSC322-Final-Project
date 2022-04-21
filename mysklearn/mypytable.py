from mysklearn import myutils

import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        num_rows = len(self.data)
        num_cols = len(self.data[0])
        return num_rows, num_cols # TODO: fix this

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        if isinstance(col_identifier, str):
            col_index = self.column_names.index(col_identifier)
        elif isinstance(col_identifier, int):
            col_index = col_identifier
        else:
            raise ValueError("ValueError exception thrown")
        col_values = []
        for row in self.data:
            if row[col_index] != "NA" or include_missing_values:
                col_values.append(row[col_index])
        return col_values # TODO: fix this

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        updated_data = []
        for row in self.data:
            updated_row = []
            for value in row:
                try:
                    value = float(value)
                except ValueError:
                    # print("CANNOT CONVERT:", value)
                    pass
                updated_row.append(value)
            updated_data.append(updated_row)
        self.data = updated_data # TODO: fix this

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        updated_data = []
        for i, row in enumerate(self.data):
            if i not in row_indexes_to_drop:
                updated_data.append(row)
        self.data = updated_data # TODO: fix this

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        # TODO: finish this
        column_names = []
        data = []

        file = open(filename)
        csvreader = csv.reader(file)
        column_names = next(csvreader)
        for row in csvreader:
            data.append(row)
        file.close()

        self.column_names = column_names
        self.data = data
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        file = open(filename, 'w')
        csvwriter = csv.writer(file)
        csvwriter.writerow(self.column_names)
        csvwriter.writerows(self.data)
        file.close()
        # TODO: fix this

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        key_column_indexes = []
        for key_column_name in key_column_names:
            key_column_indexes.append(self.column_names.index(key_column_name))
        
        dup_row_indexes = []
        for i, base_row in enumerate(self.data):
            if i not in dup_row_indexes:
                for j, subseq_row in enumerate(self.data[i+1:], start=i+1):
                    duplicate = True
                    for key_column_index in key_column_indexes:
                        if base_row[key_column_index] != subseq_row[key_column_index]:
                            duplicate = False
                    if duplicate:
                        dup_row_indexes.append(j)

        return dup_row_indexes # TODO: fix this

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        updated_data = []
        for row in self.data:
            contains_missing_value = False
            for value in row:
                if value == "NA":
                    contains_missing_value = True
            if contains_missing_value == False:
                updated_data.append(row)
        self.data = updated_data
        # TODO: fix this

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)

        running_total = 0.0
        count = 0
        for row in self.data:
            if row[col_index] != "NA":
                running_total += row[col_index]
                count += 1
        
        col_avg = running_total / count
        for i, instance in enumerate(self.data):
            if instance[col_index] == "NA":
                self.data[i][col_index] = col_avg
        # TODO: fix this

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        stats_header = ["attribute", "min", "max", "mid", "avg", "median"]
        stats_data = []

        if not self.data:
            return MyPyTable(stats_header, stats_data)

        columns = []
        for col_name in col_names:
            columns.append(self.get_column(col_name, False))
        
        for i, col in enumerate(columns):
            attr = col_names[i]
            min_value = None
            max_value = None
            running_total = 0.0
            count = 0

            for value in col:
                if min_value == None or value < min_value:
                    min_value = value
                if max_value == None or value > max_value:
                    max_value = value
                running_total += value
                count += 1
            
            mid_value = (min_value + max_value) / 2
            avg_value = running_total / count

            sorted_col_values = sorted(col)
            col_length = len(col)
            index = (col_length - 1) // 2
            if (col_length % 2):
                median_value = sorted_col_values[index]
            else:
                median_value = (sorted_col_values[index] + sorted_col_values[index + 1]) / 2
            
            stats_row = [attr, min_value, max_value, mid_value, avg_value, median_value]
            stats_data.append(stats_row)

        return MyPyTable(stats_header, stats_data) # TODO: fix this

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        new_column_names = list(self.column_names)
        new_column_names.extend(name for name in other_table.column_names if name not in new_column_names)

        new_data = []
        for row in self.data:
            row_to_add = []
            for other_table_row in other_table.data:
                match = True
                for key_col_name in key_column_names:
                    if row[self.column_names.index(key_col_name)] != other_table_row[other_table.column_names.index(key_col_name)]:
                        match = False
                if match:
                    row_to_add = list(row)
                    row_to_add.extend(value for value in other_table_row if value not in row_to_add)
                    new_data.append(row_to_add)

        return MyPyTable(new_column_names, new_data) # TODO: fix this

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        new_column_names = list(self.column_names)
        new_column_names.extend(name for name in other_table.column_names if name not in new_column_names)

        new_data = []
        for row in self.data:
            no_match_in_whole_table = True
            for other_table_row in other_table.data:
                match = True
                for key_col_name in key_column_names:
                    if row[self.column_names.index(key_col_name)] != other_table_row[other_table.column_names.index(key_col_name)]:
                        match = False
                if match:
                    row_to_add = list(row)
                    row_to_add.extend(value for value in other_table_row if value not in row_to_add)
                    new_data.append(row_to_add)
                    no_match_in_whole_table = False
            if no_match_in_whole_table:
                row_to_add = list(row)
                for i in range(len(new_column_names) - len(row_to_add)):
                    row_to_add.append(None) # correctly sizes the row
                for other_table_col_name in other_table.column_names:
                    if other_table_col_name not in self.column_names:
                        row_to_add[new_column_names.index(other_table_col_name)] = "NA"
                new_data.append(row_to_add)
        for second_table_row in other_table.data:
            no_match_in_joined_table = True
            for joined_table_row in new_data:
                match = True
                for key_column_name in key_column_names:
                    if second_table_row[other_table.column_names.index(key_column_name)] != joined_table_row[new_column_names.index(key_column_name)]:
                        match = False
                if match:
                    no_match_in_joined_table = False
            if no_match_in_joined_table:
                row_to_add = [None] * len(new_column_names) # correctly sizes the row
                for second_table_col_name in other_table.column_names:
                    row_to_add[new_column_names.index(second_table_col_name)] = second_table_row[other_table.column_names.index(second_table_col_name)]
                for first_table_col_name in self.column_names:
                    if first_table_col_name not in other_table.column_names:
                        row_to_add[new_column_names.index(first_table_col_name)] = "NA"
                new_data.append(row_to_add)

        return MyPyTable(new_column_names, new_data) # TODO: fix this