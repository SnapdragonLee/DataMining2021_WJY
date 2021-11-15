import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import input
import seaborn as sb

import pickle

import path
import tool
from bert_encode import *
from input import *


def predict():
    with open('RandomForestRegressor link s n_estimators=200.model', 'rb') as r_f_r:
        regressor = pickle.load(r_f_r)
    print('load model done')
    test_data = input.get_data(False)
    """
    encoder = BertEncoder(path.pretrained_model)
    test_encode_data = encoder.chinese2encode_bert(test_data['merged_sentences'])
    
    with open('test_dataset_bert_encode.txt', 'wb') as p_t_d_f:
        pickle.dump(test_encode_data, p_t_d_f)

    print('bert encoding finish and saved')
    """

    with open('test_dataset_bert_encode.txt', 'rb') as p_t_d_f:
        test_encode_data = pickle.load(p_t_d_f)
    print('bert encoding load')

    y_pred = regressor.predict(test_encode_data)
    print('predict done')
    y_pred = y_pred.tolist()
    format_y = []
    for line in y_pred:
        tmp_line = [str(num) for num in line]
        s = ','
        format_y.append(s.join(tmp_line))
    ids = [line[0] for line in test_data['ids']]
    with open('submit link train 400.tsv', 'w') as submit_f:
        submit_f.write('id\temotion\n')
        for i in range(len(ids)):
            submit_f.write('{0}\t{1}\n'.format(ids[i], format_y[i]))
    print('done')


def train_and_save():
    y, x = input.build_train_data()

    encoder = BertEncoder(path.pretrained_model)
    x = encoder.chinese2encode_bert(x)
    del encoder

    x = np.array(x)
    y = np.array(y)
    # 随机处理
    hstack = np.hstack((x, y))
    np.random.shuffle(hstack)
    y = hstack[:, -6:]
    x = hstack[:, :-6]

    print('bert encoding finish and shuffle')
    pre_treated_dataset = y, x
    with open(path.build_tmp_data_path('pre_treated_dataset.txt'), 'wb') as p_t_d_f:
        pickle.dump(pre_treated_dataset, p_t_d_f)

    print('bert encoding finish and saved')
    """

    with open(path.build_tmp_data_path('pre_treated_dataset.txt'), 'rb') as p_t_d_f:
        pre_treated_dataset = pickle.load(p_t_d_f)
    print("load data")

    y, x = pre_treated_dataset
    """
    l = int(len(y) * 0.8)
    y_train = y[:l]
    x_train = x[:l]
    y_test = y[l:]
    test_x = x[l:]
    link_y_test = []
    link_x_test = []
    for i in range(len(y_test)):
        if True:
            link_y_test.append(y_test[i])
            link_x_test.append(test_x[i])

    print('begin fit regressor')
    regressor = RandomForestRegressor(n_estimators=600)
    regressor.fit(X=x_train, y=y_train)

    with open(path.build_model_path('RandomForestRegressor n_estimators=600 resample.model'), 'wb') as r_f_r:
        pickle.dump(regressor, r_f_r)

    print('RandomForest fit and save')
    """

    with open(path.build_model_path('RandomForestRegressor n_estimators=200 resample.model'), 'rb') as r_f_r:
        regressor = pickle.load(r_f_r)
    print('load model')
    """
    y_pred = regressor.predict(link_x_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(link_y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(link_y_test, y_pred))
    rmse = np.sqrt(metrics.mean_squared_error(link_y_test, y_pred))
    print('Root Mean Squared Error:', rmse)
    print('Scores:', 1 / (rmse + 1))
    link_y_test = np.array(link_y_test)
    for i in range(6):
        # tool.liner_plot(link_y_test[:, i], y_pred[:, i], str(i))
        pass
    plt.show()


if __name__ == '__main__':
    train_and_save()
    # predict()
