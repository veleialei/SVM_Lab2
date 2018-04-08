from svm_basic import svm_basic
from numpy import *
from time import sleep
import matplotlib.pyplot as plt

def get_data(filename):
    datamatrix = []
    labelmatrix = []

    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        datamatrix.append([float(lineArr[0]), float(lineArr[1])])
        labelmatrix.append(int(lineArr[2]))
    return datamatrix, labelmatrix

def plot_fit(fit_line, datamatrix, labelmatrix, b):
    weights = fit_line.getA()
    dataarray = asarray(datamatrix)
    n = dataarray.shape[0]

    # Keep track of the two classes in different arrays so they can be plotted later...
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelmatrix[i]) == 1:
            xcord1.append(dataarray[i, 0]) #i, 1
            ycord1.append(dataarray[i, 1]) #i, 2
        else:
            xcord2.append(dataarray[i, 0]) #i, 1
            ycord2.append(dataarray[i, 1]) #i, 2
    fig = plt.figure()

    # Plot the data as points with different colours
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # Plot the best-fit line
    x = arange(-1.5, 6.5, 0.1)
    y = (- weights[0] * x) / weights[1] + b #descent not acsent
    ax.plot(x, y)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def accuracy(labels, hypotheses):
    count = 0.0
    correct = 0.0

    for l, h in zip(labels, hypotheses):
        count += 1.0
        if l == h:
            correct += 1.0
    return correct / count


def print_confusion_matrix(labels, hypotheses):
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    count = 1.0
    for l, h in zip(labels, hypotheses):
        count += 1.0
        if l == 1 and h == 1:
            tp += 1.0
        elif l == 1 and h == 0:
            tn += 1.0
        elif l == 0 and h == 0:
            fn += 1.0
        else:
            tn += 1
    print ('-----------------------------')
    print ('\tConfusion Matrix')
    print ('-----------------------------')
    print ('\t\tPredicted')
    print ('\tActual\tNO\tYES')
    print ('-----------------------------')
    print ('\tNO\t', tn, '\t', fp)
    print ('-----------------------------')
    print ('\tYES\t', fn, '\t', tp)
    print ('-----------------------------')

dataArr, labelArr = get_data('linearly_separable.csv')
test = svm_basic(mat(dataArr), mat(labelArr).transpose(), 1, 0.001)
b, alphas = test.smoPK(50)
w = test.fit()
w = asmatrix(w)
hypotheses = test.predict(dataArr)
print("b =", b)
print("weights =\n", w)
plot_fit(w, dataArr, labelArr, squeeze(asarray(b)))
print ('Accuracy:', accuracy(labelArr, hypotheses))
print_confusion_matrix(labelArr, hypotheses)
