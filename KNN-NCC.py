import numpy as np
from keras.datasets import cifar10
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd


def ncc(Trnx, Trny, Testx):
    clf = NearestCentroid(metric='euclidean', shrink_threshold=None)
    clf.fit(Trnx, Trny)
    y_pred = clf.predict(Testx)
    return y_pred


def knn(Trnx, Trny, Testx, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', algorithm='auto')
    knn.fit(Trnx, Trny)
    y_pred = knn.predict(Testx)
    return y_pred


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



grayscale_boolean = True
(Trnx, Trny), (Tstx, Tsty) = cifar10.load_data()
print('Train: X=%s, y=%s' % (Trnx.shape, Trny.shape))
print('Test: X=%s, y=%s' % (Tstx.shape, Tsty.shape))
Trnx = Trnx.astype('float32')
Tstx = Tstx.astype('float32')
if grayscale_boolean:
   Trnx, Tstx = grayscale(Trnx, Tstx)


Trnx = Trnx / 255.0  
Tstx = Tstx / 255.0  


Trnx = np.reshape(Trnx, (Trnx.shape[0], -1))  # (50000, 3072) 32 * 32 * 3 = 3072 
Trny = Trny.reshape(-1,)
Tstx = np.reshape(Tstx, (Tstx.shape[0], -1))  # (10000, 3072) 32 * 32 * 3 = 3072 
Tsty = Tsty.reshape(-1,)
print('New Train: X=%s, y=%s' % (Trnx.shape, Trny.shape))
print('New Test: X=%s, y=%s' % (Tstx.shape, Tsty.shape))

# --> Nearest Neighbor Classifier

grid = []
acc_array = []
index = 0
grid_pred = []
cifar10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for n_neighbors in range(1, 25):
    print("Number of neighbors is = "+str(n_neighbors)+".")
    y_pred = knn(Trnx, Trny, Tstx, n_neighbors)
    accuracy = accuracy_score(Tsty, y_pred)  
    print("Accuracy: " + str(accuracy * 100) + "%")
    acc_array.append(accuracy)
    grid.append(n_neighbors)
max_acc = max(acc_array)
for max_acc_value in range(0, len(acc_array)):
    if acc_array[max_acc_value] == max_acc:
        print(f'Max accuracy is: '+str(acc_array[max_acc_value]*100)+'% with n_neighbors = '+str(grid[index])
              +".")
        cf_matrix = confusion_matrix(Tsty, knn(Trnx, Trny, Tstx, grid[index]))
        sns.heatmap(pd.DataFrame(cf_matrix / np.sum(cf_matrix), index=cifar10_names, columns=cifar10_names),
                    annot=True, fmt='.2%', cmap='RdYlGn')
        plt.title('Confusion Matrix for '+str(grid[index])+'-Nearest Neighbor')
        plt.show()
        clas_rep_knn = classification_report(Tsty, knn(Trnx, Trny, Tstx, grid[index]),
                                             target_names=cifar10_names, output_dict=True)
        sns.heatmap(pd.DataFrame(clas_rep_knn).iloc[:-1, :].T, annot=True, fmt='.2%', cmap='RdYlGn')
        plt.title('Classification report for '+str(grid[index])+'-Nearest Neighbor')
        plt.xlabel('Metrics')
        plt.ylabel('Classes')
        plt.show()
    index += 1

# --> Nearest Centroid

y_pred_ncc = ncc(Trnx, Trny, Tstx)
accuracy_ncc = accuracy_score(Tsty, y_pred_ncc)
print("Accuracy: " + str(accuracy_ncc * 100) + "%")
cf_matrix_ncc = confusion_matrix(Tsty, y_pred_ncc)
sns.heatmap(pd.DataFrame(cf_matrix_ncc/np.sum(cf_matrix_ncc), index=cifar10_names, columns=cifar10_names),
            annot=True, fmt='.2%', cmap='RdYlGn')
plt.title('Confusion Matrix for Nearest Centroid')
plt.show()
clas_rep_ncc = classification_report(Tsty, y_pred_ncc, target_names=cifar10_names, output_dict=True)
sns.heatmap(pd.DataFrame(clas_rep_ncc).iloc[:-1, :].T, annot=True, fmt='.2%', cmap='RdYlGn')
plt.title('Classification report for Nearest Centroid')
plt.xlabel('Metrics')
plt.ylabel('Classes')
plt.show()


