import numpy as np
import pandas as pd
import sklearn.ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import const_val
import input
import seaborn as sb

import pickle

import path
import tool
from bert_encode import *
from input import *

"""
    随机森林在树木数量600出现了预测下降，最优值应该在400 ~ 500左右
"""


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


def train_and_save_regressor(model_name: str, pre_treated_dataset, have_fit=True):
    if have_fit:
        with open(path.build_model_path(model_name), 'rb') as r_f_r:
            regressor = pickle.load(r_f_r)
        print('load model')
        return regressor
    y, x = pre_treated_dataset
    l = int(len(y) * (1 - const_val.VERIFICATION_PERCENT / 100))
    y_train = y[:l]
    x_train = x[:l]
    print('begin fit regressor')
    regressor = RandomForestRegressor(n_estimators=400)
    regressor.fit(X=x_train, y=y_train)

    with open(path.build_model_path(path.build_model_path(model_name)), 'wb') as r_f_r:
        pickle.dump(regressor, r_f_r)

    print('RandomForest fit and save')
    return regressor


def judge_effect_mse(y_test, y_pred):
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', rmse)
    print('Scores:', 1 / (rmse + 1))


def read_data_encoded(dataset_name: str, have_encode=True):
    if have_encode:
        with open(path.build_tmp_data_path(dataset_name), 'rb') as p_t_d_f:
            pre_treated_dataset = pickle.load(p_t_d_f)
        print("load data encoded")
    else:
        y, x, classifier_y, label_encoder = input.build_train_data()
        encoder = BertEncoder(path.pretrained_model)
        x = encoder.chinese2encode_bert(x)
        del encoder
        x = np.array(x)
        y = np.array(y)
        classifier_y = np.array(classifier_y)
        classifier_y = classifier_y.reshape([classifier_y.shape[0], 1])
        # 随机处理
        hstack = np.hstack((x, y,classifier_y))
        np.random.shuffle(hstack)
        y = hstack[:, -7:-1]
        x = hstack[:, :-7]
        classifier_y = hstack[:, -1]

        print('bert encoding finish and shuffle')
        pre_treated_dataset = y, x, classifier_y, label_encoder
        with open(path.build_tmp_data_path(dataset_name), 'wb') as p_t_d_f:
            pickle.dump(pre_treated_dataset, p_t_d_f)
        print('bert encoding finish and saved')

    return pre_treated_dataset


def train_and_save_classifier(model_name: str, pre_treated_dataset, have_fit=True):
    if have_fit:
        with open(path.build_model_path(model_name), 'rb') as r_f_c:
            classifier = pickle.load(r_f_c)
        print('load classifier model')
        return classifier
    classifier = RandomForestClassifier(n_estimators=400)
    y, x = pre_treated_dataset
    l = int(len(y) * (1 - const_val.VERIFICATION_PERCENT / 100))
    y_train = y[:l]
    x_train = x[:l]
    print('begin fit classifier')
    classifier.fit(X=x_train, y=y_train)
    print('fit classifier finish')
    return classifier


def classifier_predict_and_judge_effect(classifier: RandomForestClassifier, pre_treated_dataset):
    y, x = pre_treated_dataset
    l = int(len(y) * (1 - const_val.VERIFICATION_PERCENT / 100))
    y_test = y[l:]
    x_test = x[l:]
    y_pred = classifier.predict(X=x_test)
    wrong = []
    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            wrong.append([y_pred[i], y_test[i]])

    print('Classifier accuracy:\t'.format(len(wrong) / len(y_pred)))
    counter = Counter(wrong)
    print('***************************')
    print(counter)
    print('***************************')


def regressor_predict_and_judge_effect(regressor: RandomForestRegressor, dataset):
    y_test, x_test = get_test_data(dataset)
    y_pred = regressor.predict(x_test)
    judge_effect_mse(y_test, y_pred)


def get_test_data(dataset):
    y, x = dataset
    l = int(len(y) * (1 - const_val.VERIFICATION_PERCENT / 100))
    y_test = y[l:]
    x_test = x[l:]
    return y_test, x_test


def main():
    # 读取数据
    y, x, classifier_y, label_encoder = read_data_encoded('pretreatedDataset.pretreatedData', False)
    regressor_data = y, x
    classifier_data = classifier_y, x
    # regressor = train_and_save_regressor('randomForestRegressor link1 400.model', regressor_data, False)
    # regressor_predict_and_judge_effect(regressor, regressor_data)
    classifier = train_and_save_classifier('randomForestClassifier link1 400.model', classifier_data, False)
    classifier_predict_and_judge_effect(classifier, classifier_data)


if __name__ == '__main__':
    main()
