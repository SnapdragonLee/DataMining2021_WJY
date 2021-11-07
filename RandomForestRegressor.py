from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import input
import seaborn as sb

import pickle
import tool
from bert_encode import *
from input import *


def main():
    encoder = BertEncoder(path.pretrained_model)
    data_set = input.get_data()
    pre_treated_dataset = encoder.stitch_characters_and_encode(data_set)

    print('bert encoding finish')

    with open('pre_treated_dataset.txt', 'wb') as p_t_d_f:
        pickle.dump(pre_treated_dataset, p_t_d_f)

    print('bert encoding finish and saved')

    """
    with open('pre_treated_dataset.txt', 'rb') as p_t_d_f:
        pre_treated_dataset = pickle.load(p_t_d_f)
    """

    y, x = pre_treated_dataset


    l = int(len(y) * 0.8)
    y_train = y[:l]
    x_train = x[:l]
    y_test = y[l:]
    test_x = x[l:]
    regressor = RandomForestRegressor(n_estimators=200)
    regressor.fit(X=x_train, y=y_train)

    with open('RandomForestRegressor.model', 'wb') as r_f_r:
        pickle.dump(regressor, r_f_r)

    print('RandomForest fit and save')

    """
    with open('RandomForestRegressor.model', 'rb') as r_f_r:
        regressor = pickle.load(r_f_r)
    """

    y_pred = regressor.predict(test_x)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    for i in range(6):
        tool.liner_plot(y_test, y_pred, str(i))

    plt.show()


if __name__ == '__main__':
    main()
