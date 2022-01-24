import math
import time
import sklearn
import tensorflow as tf
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from pandas import read_csv
from tensorflow.keras import layers
from tensorflow.keras.initializers import Initializer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.python.keras.regularizers import l2

batch_size = 64
epochs = 500
n_h2 = [32, 128, 256, 512]  # [32, 64, 128, 256]
prop = [0.25, 0.35]  # [0.2, 0.35, 0.5]
n_rbf = [5, 10, 15, 20]  # [40, 50, 75]
lr = [1e-3, 1e-4]  # [1e-3, 1e-4]
rho_array = [0.44, 0.69, 0.99]
l2_val = [l2(1e-3), l2(1e-4), l2(1e-5)]
a_val = [1e-3, 1e-4, 1e-5]


class RBFLayer(layers.Layer):
    def __init__(self, output_dim, initializer, betas=1.0, **kwargs):
        self.betas = betas
        self.output_dim = output_dim
        self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers', shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer, trainable=True)
        d_max = 0
        for i in range(0, self.output_dim):
            for j in range(0, self.output_dim):
                d = np.linalg.norm(self.centers[i] - self.centers[j])
                if d > d_max:
                    d_max = d
        sigma = d_max / np.sqrt(2 * self.output_dim)
        self.betas = np.ones(self.output_dim) / (2 * (sigma ** 2))
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        C = tf.expand_dims(self.centers, -1)  # εισάγουμε μια διάσταση από άσσους
        H = tf.transpose(C - tf.transpose(inputs))  # Πίνακας με τις διαφορές
        return tf.exp(-self.betas * tf.math.reduce_sum(H ** 2, axis=1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class InitCentersKMeans(Initializer):
    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter
        super().__init__()

    def __call__(self, shape, dtype=None, *args):
        assert shape[1] == self.X.shape[1]
        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def data_pre_processing():
    dataset = read_csv('insurance.csv')

    le = LabelEncoder()
    dataset['sex'] = le.fit_transform(dataset['sex'])
    dataset['smoker'] = le.fit_transform(dataset['smoker'])
    dataset['region'] = le.fit_transform(dataset['region'])
    X, y = dataset.values[:, :-1], dataset.values[:, -1]
    Trnx, Tstx, Trny, Tsty = train_test_split(X, y, test_size=0.4, shuffle=True)
    print('Train: X=%s, y=%s' % (Trnx.shape, Trny.shape))
    print('Test: X=%s, y=%s' % (Tstx.shape, Tsty.shape))
    Trny = Trny.reshape(-1, 1)
    Tsty = Tsty.reshape(-1, 1)
    scalerx = StandardScaler()
    scalery = StandardScaler()
    Trnx = scalerx.fit_transform(Trnx)
    Trny = scalery.fit_transform(Trny)
    Tsty = scalery.transform(Tsty)
    Tstx = scalerx.transform(Tstx)
    print(dataset.head())
    print('Train: X=%s, y=%s' % (Trnx.shape, Trny.shape))
    print('Test: X=%s, y=%s' % (Tstx.shape, Tsty.shape))
    x_full = tf.concat([Trnx, Tstx], 0).numpy()
    y_full = tf.concat([Trny, Tsty], 0).numpy()

    return Trnx, Trny, Tstx, Tsty, x_full, y_full


# def print_plots(lr, loss, metric1, metric2):
#     model = model_rbf(Trnx, Trny, Tstx, Tsty, lr, loss, metric1, metric2)
#     history = model.fit(Trnx, Trny, batch_size=batch_size, epochs=epochs, validation_data=(Tstx, Tsty))
#     # score = model.evaluate(Tstx, Tsty)
#
#     # trn_loss = [history.history['loss'][i] for i in range(100)]
#     # val_loss = [history.history['val_loss'][i] for i in range(100)]
#     #
#     # trn_r2 = history.history['r2_score']
#     # val_r2 = history.history['val_r2_score']
#
#     trn_rmse = history.history['rmse']
#     val_rmse = history.history['val_rmse']
#
#     # plt.plot(trn_loss)
#     # plt.plot(val_loss)
#     #
#     # plt.title('Learning Curve of training dataset and lr=0.001')
#     # plt.xlabel('Epochs')
#     # plt.ylabel('Loss')
#     # plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
#     # plt.savefig('RBF_loss.png')
#     # plt.close()
#     #
#     # plt.plot(trn_r2)
#     # plt.plot(val_r2)
#     #
#     # plt.title('R2 of training dataset and lr=0.001')
#     # plt.xlabel('Epochs')
#     # plt.ylabel('R2')
#     # plt.legend(['Training R2', 'Validation R2'], loc='lower right')
#     # plt.savefig('RBF_R2.png')
#     # plt.close()
#
#     plt.plot(trn_rmse)
#     plt.plot(val_rmse)
#
#     plt.title('RMSE of training dataset and lr=0.001')
#     plt.xlabel('Epochs')
#     plt.ylabel('RMSE')
#     plt.legend(['Training RMSE', 'Validation RMSE'], loc='lower right')
#     plt.savefig('RBF_RMSE.png')
#     plt.close()


def model_rbf(n, n_dense, lr, p, rho, loss, metric1):
    Trnx, Trny, Tstx, Tsty, _, _ = data_pre_processing()
    model = Sequential()
    model.add(RBFLayer(n, initializer=InitCentersKMeans(Trnx), input_shape=(6,)))
    model.add(Dense(n_dense, activation='relu', kernel_regularizer=l2(1e-5)))
    # model.add(Dropout(p))
    model.add(Dense(1))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,
                                                                 decay_steps=10000,
                                                                 decay_rate=0.9)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=rho), loss=loss, metrics=[metric1])
    history = model.fit(Trnx, Trny, batch_size=batch_size, epochs=epochs, validation_data=(Tstx, Tsty))
    y_pred = model.predict(Tstx)
    # for i in range(0, y_pred.shape[0]):
    #     max_value = max(y_pred[i])
    #     for j in range(0, y_pred.shape[1]):
    #         if y_pred[i][j] == max_value:
    #             y_pred[i][j] = 1
    #         else:
    #             y_pred[i][j] = 0
    # print(Tsty[0])
    # print(y_pred[0])
    for i in range(0, Tsty.shape[0]):
        boolean = tf.reduce_all(tf.math.equal(Tsty[i], y_pred[i]))
        if not boolean:
            print('Real value :'+str(Tsty[i]))
            print('Predicted value: '+str(y_pred[i]))
            break
    trn_loss = [history.history['loss'][i] for i in range(epochs)]
    val_loss = [history.history['val_loss'][i] for i in range(epochs)]
    plt.plot(trn_loss)
    plt.plot(val_loss)

    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Testing Loss'], loc='upper right')
    plt.savefig('RBF_loss.png')
    plt.close()
    trn_RMSE = history.history['rmse']
    val_RMSE= history.history['val_rmse']

    plt.plot(val_RMSE)
    plt.plot(trn_RMSE)

    plt.title('RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend(['Testing RMSE', 'Training RMSE'], loc='upper right')
    plt.savefig('RBF_RMSE.png')
    plt.close()
    return model


def grid_search_rmse():

    times = []  # Φόρτωση χρόνου
    k = 5  # 5-fold cross val
    grid_rmse = []
    index = 0
    start = time.time()
    rmse_f = []
    _, _, _, _, x, y = data_pre_processing()
    for n in n_rbf:
        for nh2 in n_h2:
            for lear_rate in lr:
                for rho in rho_array:
                    for a in range(0, len(a_val)):
                        Rmse = []
                        kf = KFold(5, shuffle=True, random_state=42)  # Use for KFold classification with random_state = 42
                        fold = 0
                        for train, test in kf.split(x):
                            fold += 1
                            print(f'Our fold is: ' + str(fold) + '/' + str(k) + ' with: n_h2=' + str(
                                nh2) + ', for n_centers ' + str(n) + ' for rho ' + str(rho) + ' a=' + str(a_val[a]) +
                                  ' and learning_rate=' + str(lear_rate))
                            x_train = x[train]
                            y_train = y[train]
                            x_test = x[test]
                            y_test = y[test]
                            model = Sequential()
                            model.add(RBFLayer(n, initializer=InitCentersKMeans(x_train), input_shape=(6,)))
                            model.add(Dense(nh2, activation='relu', kernel_regularizer=l2_val[a]))
                            model.add(Dense(1))
                            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lear_rate,
                                                                                         decay_steps=10000,
                                                                                         decay_rate=0.9)
                            model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=rho),
                                          loss=mse, metrics=rmse)
                            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                                validation_data=(x_test, y_test))

                            Rmse.append(min(history.history['val_rmse']))

                        rmse_f.append(sum(Rmse) / len(Rmse))
                        print(f'RMSE is = ' + str(rmse_f[-1]))
                        grid_rmse.append((n, nh2, lear_rate, rho, a_val[a]))
    times.append((time.time() - start))
    print(f'It lasted ' + str(times[0]) + ' secs')
    # Εύρεση του ελαχίστου rmse
    min_val_of_rmse_f = min(rmse_f)
    for rmse_value in range(0, len(rmse_f)):
        if rmse_f[rmse_value] == min_val_of_rmse_f:
            print(f'Min RMSE is ' + str(rmse_f[rmse_value]) + ' with: n_h2=' + str(grid_rmse[index][1]) +
                  ' for n_centers ' + str(grid_rmse[index][0]) + ' a=' + str(grid_rmse[index][4]) +
                  ' for rho ' + str(grid_rmse[index][3]) + ' and learning_rate='+str(grid_rmse[index][2]))
        index += 1


def grid_search_r2():
    times = []  # Φόρτωση χρόνου
    k = 5  # 5-fold cross val
    grid_r2 = []
    index_1 = 0
    start = time.time()
    R2_score_f = []
    _, _, _, _, x, y = data_pre_processing()
    for n in n_rbf:
        for nh2 in n_h2:
            for lear_rate in lr:
                for p in prop:
                    for rho in rho_array:
                        for a in range(0, len(a_val)):
                            R2_score = []
                            kf = KFold(5, shuffle=True, random_state=42)  # KFold classification with random_state = 42
                            fold = 0
                            for train, test in kf.split(x):
                                fold += 1
                                print(f'Our fold is: '+str(fold)+'/'+str(k)+' with: n_h2='+str(nh2)+', for n_centers ' +
                                      str(n)+' for prop='+str(p)+' for rho '+str(rho)+' a=' + str(a_val[a]) +
                                      ' and learning_rate='+str(lear_rate))
                                x_train = x[train]
                                y_train = y[train]
                                x_test = x[test]
                                y_test = y[test]
                                model = Sequential()
                                model.add(RBFLayer(n, initializer=InitCentersKMeans(x_train), input_shape=(6,)))
                                model.add(Dense(nh2, activation='relu', kernel_regularizer=l2_val[a]))
                                model.add(Dropout(p))
                                model.add(Dense(1))
                                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                    initial_learning_rate=lear_rate, decay_steps=10000, decay_rate=0.9)
                                model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, rho=rho),
                                              loss=mse, metrics=r2_score)
                                history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                                    validation_data=(x_test, y_test))
                                R2_score.append(max(history.history['val_r2_score']))

                            R2_score_f.append(sum(R2_score) / len(R2_score))
                            print(f'R2 is = ' + str(R2_score_f[-1]))
                            grid_r2.append((n, nh2, lear_rate, rho, p, a_val[a]))
    times.append((time.time() - start))
    print(f'It lasted ' + str(times[0]) + ' secs')
    # Εύρεση του μεγίστου R2
    max_val_of_r2_f = max(R2_score_f)
    for R2_value in range(0, len(R2_score_f)):
        if R2_score_f[R2_value] == max_val_of_r2_f:
            print(f'Max R2 is ' + str(R2_score_f[R2_value])+' with: n_h2=' + str(grid_r2[index_1][1]) +
                  ' for n_centers ' + str(grid_r2[index_1][0]) + ' for prop='+str(grid_r2[index_1][4]) +
                  ' a=' + str(grid_r2[index_1][5])+' for rho ' + str(grid_r2[index_1][3]) +
                  ' and learning_rate='+str(grid_r2[index_1][2]))
        index_1 += 1


def knn(Trnx, Trny, Testx, Tsty, i_start, i_finish):
    for i in range(i_start, i_finish):
        knn = KNeighborsRegressor(n_neighbors=i)
        knn.fit(Trnx, Trny)
        y_pred = knn.predict(Testx)
        print("r2 for KNN n="+str(i)+" : =" + str(sklearn.metrics.r2_score(Tsty, y_pred)))
        print("RMSE for KNN n="+str(i)+" : =" + str(math.sqrt(sklearn.metrics.mean_squared_error(Tsty, y_pred))))


def main():

    grid_search_rmse()
    # model_rbf(n=10, n_dense=512, lr=3e-4, p=0.25, rho=0.69, loss=mse, metric1=rmse)


if __name__ == '__main__':
    main()
