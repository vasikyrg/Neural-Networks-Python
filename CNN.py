import time
import numpy as np
import pandas as pd
from keras.datasets import cifar10
import seaborn as sns
import tensorflow as tf
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from tensorflow.keras.regularizers import l2

epochs = 50
batch_size = 256
l2_val = [l2(0.001), l2(0.0001), l2(0.000001)]
a_val = [0.001, 0.0001, 0.000001]
lr_val = [0.001, 0.0001, 0.00001]
k = 5
grayscale_boolean = False


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


def model_cifar_CNN(first_layer_nodes, second_layer_nodes, third_layer_nodes):
    model = Sequential()
    model.add(Conv2D(first_layer_nodes, (3, 3), padding='same', activation='relu',
                     kernel_initializer=tf.keras.initializers.HeNormal(),
                     input_shape=Trnx.shape[1:], kernel_regularizer=l2(1e-3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv2D(first_layer_nodes, (3, 3), padding='same', activation='relu',
                     kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=l2(1e-3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(second_layer_nodes, (3, 3), padding='same', activation='relu',
                     kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=l2(1e-3)))
    model.add(BatchNormalization())
    model.add(Conv2D(second_layer_nodes, (3, 3), padding='same', activation='relu',
                     kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=l2(1e-3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(third_layer_nodes, (3, 3), padding='same', activation='relu',
                     kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=l2(1e-3)))
    model.add(BatchNormalization())
    model.add(Conv2D(third_layer_nodes, (3, 3), padding='same', activation='relu',
                     kernel_initializer=tf.keras.initializers.HeNormal(), kernel_regularizer=l2(1e-3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def print_graphs(first_layer_nodes, second_layer_nodes, third_layer_nodes):
    model = model_cifar_CNN(first_layer_nodes, second_layer_nodes, third_layer_nodes)
    history = model.fit(Trnx, Trny, batch_size=batch_size, epochs=epochs, validation_data=(Tstx, Tsty))
    y_pred = model.predict(Tstx)
    cifar10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    trn_acc = [history.history['accuracy'][i] * 100
               for i in range(len(history.history['accuracy']))]
    val_acc = [history.history['val_accuracy'][i] * 100
               for i in range(len(history.history['val_accuracy']))]

    trn_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(trn_loss)
    plt.plot(val_loss)

    plt.title('Losses ')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Losses', 'Validation Losses'], loc='upper right')
    plt.show()
    plt.close()

    plt.plot(trn_acc)
    plt.plot(val_acc)

    plt.title('Accuracy ')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
    plt.show()
    plt.close()

    cf_matrix = confusion_matrix(Tsty.argmax(axis=1), y_pred.argmax(axis=1))
    sns.heatmap(pd.DataFrame(cf_matrix / np.sum(cf_matrix), index=cifar10_names, columns=cifar10_names),
                annot=True, fmt='.2%', cmap='RdYlGn')
    plt.title('Confusion Matrix for MLP Cifar-10 dataset')
    plt.show()

    clas_rep_knn = classification_report(Tsty.argmax(axis=1), y_pred.argmax(axis=1),
                                         target_names=cifar10_names, output_dict=True)
    sns.heatmap(pd.DataFrame(clas_rep_knn).iloc[:-1, :].T, annot=True, fmt='.2%', cmap='RdYlGn')
    plt.title('Classification report for MLP Cifar-10 dataset')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.show()


def grid_search_manual_with_cv(a_val, lr_val):

    loss = []
    loss_f = []
    grid = []
    start = time.time()
    index = 0
    times = []

    for a in range(0, len(a_val)):
        for lr in range(0, len(lr_val)):
            print(f'At this moment a=' + str(a_val[a]) + ' and lr=' + str(lr_val[lr]))
            # Grid Search
            f_measures = []
            kf = KFold(5, shuffle=True, random_state=42)  # Use for KFold classification with random_state = 42
            fold = 0
            for train, test in kf.split(x):
                fold += 1
                print(
                    f'Our fold is: '+str(fold)+'/'+str(k)+' for a=' + str(a_val[a]) + ' and lr=' + str(lr_val[lr]))
                
                x_train = x[train]
                y_train = y[train]
                x_val = x[test]
                y_val = y[test]
                
                model = Sequential()
                model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                                 kernel_initializer=tf.keras.initializers.HeNormal(),
                                 input_shape=Trnx.shape[1:], kernel_regularizer=l2_val[a]))
                model.add(BatchNormalization())
                model.add(Dropout(0.2))
                model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                                 kernel_initializer=tf.keras.initializers.HeNormal(),
                                 kernel_regularizer=l2_val[a]))
                model.add(BatchNormalization())
                model.add(MaxPooling2D((2, 2)))
                model.add(Dropout(0.2))
                model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
                                 kernel_initializer=tf.keras.initializers.HeNormal(),
                                 kernel_regularizer=l2_val[a]))
                model.add(BatchNormalization())
                model.add(MaxPooling2D((2, 2)))
                model.add(Dropout(0.3))
                model.add(Flatten())
                model.add(Dense(10, activation='softmax'))
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_val[lr])
                model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                    validation_data=(x_val, y_val))
                loss.append(max(history.history['val_accuracy']))
            loss_f.append(sum(loss) / len(loss))
            grid.append((a_val[a], lr_val[lr]))
    times.append((time.time() - start))
    print(f'It lasted ' + str(times[0]) + ' secs')
    # Εύρεση του μεγίστου f_measure
    max_val_of_loss = max(loss_f)
    for loss_value in range(0, len(loss_f)):
        if loss_f[loss_value] == max_val_of_loss:
            print(f'Max Accuracy is ' + str(loss_f[loss_value]) +
                  ' with: a=' + str(grid[index][0]) + ' lr=' + str(grid[index][1]) + '')
        index += 1



(Trnx, Trny), (Tstx, Tsty) = cifar10.load_data()


Trny = np_utils.to_categorical(Trny, 10)
Tsty = np_utils.to_categorical(Tsty, 10)


Trnx = Trnx.astype('float32')
Tstx = Tstx.astype('float32')


if grayscale_boolean:
   Trnx, Tstx = grayscale(Trnx, Tstx)


Trnx = Trnx / 255.0  
Tstx = Tstx / 255.0  


x = tf.concat([Trnx, Tstx], 0).numpy()
y = tf.concat([Trny, Tsty], 0).numpy()

print_graphs(32, 64, 128)

