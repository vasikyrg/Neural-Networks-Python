import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow import keras
import time
import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


def grayscale(Trnx, Tstx):
    output_list_trnx = []
    output_list_tstx = []
    for i in range(Trnx.shape[0]):
        output_list_trnx.append(tf.image.rgb_to_grayscale(Trnx[i]))
    Trnx_Gray = tf.stack(output_list_trnx)
    # Μετατροπή του Testx
    for i in range(Tstx.shape[0]):
        output_list_tstx.append(tf.image.rgb_to_grayscale(Tstx[i]))
    Tstx_Gray = tf.stack(output_list_tstx)
    print('Train after Grayscale X=%s: ' % Trnx_Gray.shape)
    print('Test after Grayscale X=%s: ' % Tstx_Gray.shape)
    return Trnx_Gray, Tstx_Gray


def conf_matrix(tsty, y_pred):
    cifar10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cf_matrix = confusion_matrix(tsty, y_pred)
    sns.heatmap(pd.DataFrame(cf_matrix / np.sum(cf_matrix), index=cifar10_names, columns=cifar10_names),
                annot=True, fmt='.2%', cmap='RdYlGn')
    plt.title('Confusion Matrix for SVM')
    plt.show()


def data_pre_processing(grayscale_boolean, normalized_boolean, train_shape, test_shape):
    (Trnx, Trny), (Tstx, Tsty) = keras.datasets.cifar10.load_data()
    print('Train: X=%s, y=%s' % (Trnx.shape, Trny.shape))
    print('Test: X=%s, y=%s' % (Tstx.shape, Tsty.shape))
    Trnx = Trnx.astype('float32')
    Tstx = Tstx.astype('float32')
    if grayscale_boolean:
        Trnx, Tstx = grayscale(Trnx, Tstx)
    if normalized_boolean:
        Trnx = ((Trnx / 255.0) * 2) - 1  
        Tstx = ((Tstx / 255.0) * 2) - 1  
    Trnx = Trnx[:train_shape, :]
    Trny = Trny[:train_shape]
    Tstx = Tstx[:test_shape, :]
    Tsty = Tsty[:test_shape]

    Trnx = np.reshape(Trnx, (Trnx.shape[0], -1))  # (50000, 3072) 32 * 32 * 3 = 3072
    Trny = Trny.reshape(-1, )
    Tstx = np.reshape(Tstx, (Tstx.shape[0], -1))  # (10000, 3072) 32 * 32 * 3 = 3072
    Tsty = Tsty.reshape(-1, )
    print('New Train: X=%s, y=%s' % (Trnx.shape, Trny.shape))
    print('New Test: X=%s, y=%s' % (Tstx.shape, Tsty.shape))
    return Trnx, Trny, Tstx, Tsty


def poly_kernel():
    Trnx, Trny, Tstx, Tsty = data_pre_processing(grayscale_boolean=True, normalized_boolean=False,
                                                 train_shape=10000, test_shape=9000)
    C = [1, 0.1, 100, 10]  # [1, 0.1, 100, 10]
    gamma = ['scale', 'auto', 0.1, 10]  # [1, 0.1, 'auto', 10]
    kern = 'poly'  # ['rbf', 'sigmoid', 'linear', 'poly']
    times = []  # Φόρτωση χρόνου
    pred_array = []
    grid = []
    start = time.time()
    for g in gamma:
        for c in C:
            for deg in range(2, 5):
                clf = svm.SVC(kernel=kern, degree=deg, C=c, gamma=g, decision_function_shape='ovr',
                              cache_size=512).fit(Trnx, Trny)
                pred = clf.predict(Tstx)
                print("(" + str(kern) + ") Accuracy for C=" + str(c) + ", degree=" + str(deg) + ", gamma=" + str(g) +
                      " was: " + str(accuracy_score(Tsty, pred)))
                pred_array.append(accuracy_score(Tsty, pred))
                grid.append([c, g, kern, deg, pred])
    times.append((time.time() - start))
    print(f'It lasted ' + str(times[0]) + ' secs')
    max_value_of_array = max(pred_array)
    index = 0
    for max_value in pred_array:
        if max_value == max_value_of_array:
            print(f'Max ' + str(grid[index][2]) + ' value is: ' + str(max_value * 100) + '% with c=' +
                  str(grid[index][0]) + ', gamma=' + str(grid[index][1]) +
                  ' and degree=' + str(grid[index][3]))
            conf_matrix(Tsty, grid[index][4])
            return max_value
        index += 1


def rbf_kernel():
    Trnx, Trny, Tstx, Tsty = data_pre_processing(grayscale_boolean=False, normalized_boolean=True,
                                                 train_shape=50000, test_shape=10000)
    C = [1, 0.1, 100, 10]  # [1, 0.1, 100, 10]
    gamma = ['scale', 'auto', 0.1, 10]  # ['scale', 'auto', 0.1, 10]
    kern = 'rbf'  # ['rbf', 'sigmoid', 'linear', 'poly']
    times = []  # Φόρτωση χρόνου
    pred_array = []
    grid = []
    start = time.time()
    for g in gamma:
        for c in C:
            clf = svm.SVC(kernel=kern, C=c, gamma=g, decision_function_shape='ovr',
                          cache_size=512).fit(Trnx, Trny)
            pred = clf.predict(Tstx)
            print("(" + str(kern) + ") Accuracy for C=" + str(c) + ", gamma=" + str(g) +
                  " was: " + str(accuracy_score(Tsty, pred)))
            pred_array.append(accuracy_score(Tsty, pred))
            grid.append([c, g, kern, pred])
    times.append((time.time() - start))
    print(f'It lasted ' + str(times[0]) + ' secs')
    max_value_of_array = max(pred_array)
    index = 0
    for max_value in pred_array:
        if max_value == max_value_of_array:
            print(f'Max ' + str(grid[index][2]) + ' value is: ' + str(max_value * 100) + '% with c=' +
                  str(grid[index][0]) + ', gamma=' + str(grid[index][1]))
            conf_matrix(Tsty, grid[index][3])
            return max_value
        index += 1


def sigmoid_kernel():
    Trnx, Trny, Tstx, Tsty = data_pre_processing(grayscale_boolean=False, normalized_boolean=True,
                                                 train_shape=10000, test_shape=9000)
    C = [1, 0.1, 100, 10]  # [1, 0.1, 100, 10]
    gamma = ['scale', 'auto', 0.1, 10]  # [1, 0.1, 'auto', 10]
    kern = 'sigmoid'  # ['rbf', 'sigmoid', 'linear', 'poly']
    times = []  # Φόρτωση χρόνου
    pred_array = []
    grid = []
    start = time.time()
    for g in gamma:
        for c in C:
            clf = svm.SVC(kernel=kern, C=c, gamma=g, decision_function_shape='ovr',
                          cache_size=512).fit(Trnx, Trny)
            pred = clf.predict(Tstx)
            print("(" + str(kern) + ") Accuracy for C=" + str(c) + ", gamma=" + str(g) +
                  " was: " + str(accuracy_score(Tsty, pred)))
            pred_array.append(accuracy_score(Tsty, pred))
            grid.append([c, g, kern, pred])
    times.append((time.time() - start))
    print(f'It lasted ' + str(times[0]) + ' secs')
    max_value_of_array = max(pred_array)
    index = 0
    for max_value in pred_array:
        if max_value == max_value_of_array:
            print(f'Max ' + str(grid[index][2]) + ' value is: ' + str(max_value * 100) + '% with c=' +
                  str(grid[index][0]) + ', gamma=' + str(grid[index][1]))
            conf_matrix(Tsty, grid[index][3])
            return max_value
        index += 1


def linear_kernel():
    Trnx, Trny, Tstx, Tsty = data_pre_processing(grayscale_boolean=True, normalized_boolean=False,
                                                 train_shape=10000, test_shape=9000)
    kern = 'linear'  # ['rbf', 'sigmoid', 'linear', 'poly']
    times = []  # Φόρτωση χρόνου
    C = [1, 0.1, 100, 10]  # [1, 0.1, 100, 10]
    pred_array = []
    grid = []
    start = time.time()
    for c in C:
        clf = svm.SVC(kernel=kern, C=c, decision_function_shape='ovr', cache_size=512).fit(Trnx, Trny)
        pred = clf.predict(Tstx)
        print("(linear) Accuracy for C=" + str(c)+" was: " + str(accuracy_score(Tsty, pred)))
        pred_array.append(accuracy_score(Tsty, pred))
        grid.append([kern, pred])
    times.append((time.time() - start))
    print(f'It lasted ' + str(times[0]) + ' secs')
    max_value_of_array = max(pred_array)
    index = 0
    for max_value in pred_array:
        if max_value == max_value_of_array:
            print(f'Max ' + str(grid[index][0]) + ' value is: ' + str(max_value * 100))
            conf_matrix(Tsty, grid[index][1])
            return max_value
        index += 1


def main():
    # poly_kernel()
    rbf_kernel()
    # sigmoid_kernel()
    # linear_kernel()


if __name__ == '__main__':
    main()
