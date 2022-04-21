import matplotlib.pyplot as plt

from mysklearn import myutils

def draw_frequency_diagram(title, x_axis_label, y_axis_label, categories, frequencies, tilt_xticks=True):
    plt.figure(figsize=(15,8))
    plt.bar(categories, frequencies)
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if tilt_xticks:
        plt.xticks(rotation=45)
    plt.grid()
    plt.show()

def draw_pie_chart(title, labels, data):
    plt.figure(figsize=(15,8))
    plt.pie(data, labels=labels, autopct="%1.1f%%")
    plt.title(title)
    plt.show()

def draw_histogram(title, x_axis_label, y_axis_label, data, num_bins):
    plt.figure(figsize=(15,8))
    plt.hist(data, bins=num_bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.show()

def draw_scatter_plot(title, x_axis_label, y_axis_label, x_data, y_data, text_x, text_y):
    plt.figure(figsize=(15,8))
    plt.scatter(x_data, y_data)
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.grid()

    m, b = myutils.compute_slope_intercept(x_data, y_data)
    plt.plot([min(x_data), max(x_data)], [m * min(x_data) + b, m * max(x_data) + b], c="r")

    r = myutils.compute_correlation_coefficient(x_data, y_data)
    cov = myutils.compute_covariance(x_data, y_data)
    plt.annotate("corr: %.2f, cov: %.2f" %(r, cov),
        xy=(text_x, text_y), xycoords='axes fraction', color="r")
    
    plt.show()

def draw_box_plot(title, x_axis_label, y_axis_label, labels, distributions):
    plt.figure(figsize=(25,12))
    plt.boxplot(distributions)
    plt.xticks(list(range(1, len(distributions) + 1)), labels, rotation=45)
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.grid()
    plt.show()