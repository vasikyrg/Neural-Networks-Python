import time
import numpy as np
import pandas as pd
from keras.datasets import cifar10
import seaborn as sns
import tensorflow as tf
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from tensorflow.keras.regularizers import l2
from keras import backend as K


epochs = 100
batch_size = 256
size_of_hidden_up = [512, 256]
size_of_hidden_down = [256, 128]
l2_val = [l2(0.001), l2(0.0001), l2(0.000001)]
a_val = [0.001, 0.0001, 0.000001]
lr_val = [0.001, 0.0001, 0.00001]
k = 5  # 5-fold cross val
grayscale_boolean = False


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def model_cifar(first_layer_nodes, last_layer_nodes):
    model = Sequential()
    model.add(Flatten(input_shape=Trnx.shape[1:]))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(1e-4),
                    kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(BatchNormalization())
    model.add(Dense(first_layer_nodes, activation='relu', kernel_regularizer=l2(1e-4),
                    kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(BatchNormalization())
    model.add(Dense(last_layer_nodes, activation='relu', kernel_regularizer=l2(1e-4),
                    kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,
                                                                 decay_steps=10000,
                                                                 decay_rate=0.9)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


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


def print_graphs(first_layer_nodes, last_layer_nodes):
    model = model_cifar(first_layer_nodes, last_layer_nodes)
    history = model.fit(Trnx, Trny, batch_size=batch_size, epochs=epochs, validation_data=(Tstx, Tsty))
    y_pred = model.predict(Tstx)
    for i in range(0, y_pred.shape[0]):
        max_value = max(y_pred[i])
        for j in range(0, y_pred.shape[1]):
            if y_pred[i][j] == max_value:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0
    print(Tsty[0])
    print(y_pred[0])
    for i in range(0, Tsty.shape[0]):
        boolean = tf.reduce_all(tf.math.equal(Tsty[i], y_pred[i]))
        if not boolean:
            print(Tsty[i])
            print(y_pred[i])
            plt.imshow(Trnx[i])
            plt.show()
            break
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


def grid_search_manual(size_of_hidden_up, size_of_hidden_down):
    start = time.time()
    times = []  # Φόρτωση χρόνου
    grid = []
    mse = []
    mse_f = []
    index = 0
    for i in range(0, len(size_of_hidden_up)):
        for j in range(0, len(size_of_hidden_down)):
            model = model_cifar(size_of_hidden_up[i], size_of_hidden_down[j])
            history = model.fit(Trnx, Trny, batch_size=batch_size, epochs=epochs, validation_data=(Tstx, Tsty))
            mse.append(min(history.history['loss']))
            mse_f.append(sum(mse) / len(mse))
            grid.append((size_of_hidden_up[i], size_of_hidden_down[j]))
    times.append((time.time() - start))
    print(f'It lasted ' + str(times[0]) + ' secs')
    min_val_of_loss_f = min(mse_f)
    for mse_value in range(0, len(mse_f)):
        if mse_f[mse_value] == min_val_of_loss_f:
            print(f'Min MSE is ' + str(mse_f[mse_value]) + ' with: n_h2=' + str(grid[index][0]) + ' n_h3=' + str(
                grid[index][1]) + '.')
            print_graphs(grid[index][0], grid[index][1])
        index += 1


def grid_search_manual_with_cv(size_of_hidden_up, size_of_hidden_down, a_val, lr_val):
    fold = 0
    loss_f = []
    grid = []
    start = time.time()
    index = 0
    times = []
    for nh1 in range(0, len(size_of_hidden_up)):
        for nh2 in range(0, len(size_of_hidden_down)):
            for a in range(0, len(a_val)):
                for lr in range(0, len(lr_val)):
                    print(f'At this moment n_h1=' + str(size_of_hidden_up[nh1]) + ' n_h2=' +
                          str(size_of_hidden_down[nh2]) + ' a=' + str(a_val[a]) + ' and lr=' + str(lr_val[lr]))
                    # Grid Search
                    loss = []
                    kf = KFold(5, shuffle=True, random_state=42)  # Use for KFold classification with random_state = 42
                    fold = 0
                    for train, test in kf.split(x):
                        fold += 1
                        print(
                            f'Our fold is: ' + str(fold) + '/5 for n_h1=' + str(size_of_hidden_up[nh1]) +
                            ' n_h2='+str(size_of_hidden_down[nh2])+' a=' + str(a_val[a]) + ' and lr=' + str(lr_val[lr]))
                        x_train = x[train]
                        y_train = y[train]
                        x_val = x[test]
                        y_val = y[test]

                        
                        model = Sequential()
                        model.add(Flatten(input_shape=Trnx.shape[1:]))
                        model.add(BatchNormalization())
                        model.add(Dense(1024, activation='relu', kernel_regularizer=l2_val[a],
                                        kernel_initializer=tf.keras.initializers.HeNormal()))
                        model.add(BatchNormalization())
                        model.add(Dense(size_of_hidden_up[nh1], activation='relu', kernel_regularizer=l2_val[a],
                                        kernel_initializer=tf.keras.initializers.HeNormal()))
                        model.add(BatchNormalization())
                        model.add(Dense(size_of_hidden_down[nh2], activation='relu', kernel_regularizer=l2_val[a],
                                        kernel_initializer=tf.keras.initializers.HeNormal()))
                        model.add(BatchNormalization())
                        model.add(Dense(10, activation='softmax'))
                        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_val[lr],
                                                                                     decay_steps=10000,
                                                                                     decay_rate=0.8)
                        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
                        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                            validation_data=(x_val, y_val))
                        loss.append(max(history.history['val_accuracy']))
                    loss_f.append(sum(loss) / len(loss))
                    grid.append((size_of_hidden_up[nh1], size_of_hidden_down[nh2], a_val[a], lr_val[lr]))
    times.append((time.time() - start))
    print(f'It lasted ' + str(times[0]) + ' secs')
    # Εύρεση του μεγίστου validation accuracy
    max_val_of_loss = max(loss_f)
    for loss_value in range(0, len(loss_f)):
        if loss_f[loss_value] == max_val_of_loss:
            print(f'Max Accuracy is ' + str(loss_f[loss_value]) +
                  ' with: n_h1=' + str(grid[index][0]) + ' n_h2=' + str(grid[index][1]) +
                  ' a=' + str(grid[index][2]) + ' lr=' + str(grid[index][3]) + '')
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
# print('New Train: X=%s, y=%s' % (Trnx.shape, Trny.shape))
# print('New Test: X=%s, y=%s' % (Tstx.shape, Tsty.shape))
# grid_seach_auto()
# grid_search_manual(size_of_hidden_up, size_of_hidden_down)
# grid_search_manual_with_cv(size_of_hidden_up, size_of_hidden_down, a_val, lr_val)
