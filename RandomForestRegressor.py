import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import input
import seaborn as sb
import pandas as pd
from bert_encode import *
from input import *


def main():

    encoder = BertEncoder(path.pretrained_model)
    data_set = input.get_data()
    y, x = encoder.stitch_characters_and_encode(data_set)
    df = pd.DataFrame(data=[y, x], index=['y', 'x'])
    print(df)
    df.to_csv('encode.csv')
    print('encode done')

    encode_data = pd.read_csv('encode.csv', index_col=0)
    pass
    encode_data = encode_data.to_dict('index')
    x = []
    y = []

    for i in encode_data['x'].keys():
        x.append(np.array([float(num) for num in encode_data['x'][i][1:-1].split()]))
        y.append(np.array([float(num) for num in encode_data['y'][i][1:-1].split(', ')]))

    l = int(len(y) * 0.8)
    y_train = y[:l]
    x_train = x[:l]
    y_test = y[l:]
    test_x = x[l:]
    regressor = RandomForestRegressor(n_estimators=200)
    regressor.fit(X=x, y=y)
    y_pred = regressor.predict(test_x)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    sb.pairplot(y_pred, y_test)
    plt.show()


if __name__ == '__main__':
    main()
