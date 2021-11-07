import matplotlib.pyplot as plt


def liner_plot(ans, predictions, figure_name):
    plt.figure(figure_name)
    plt.scatter(ans, predictions)
    plt.xlabel('True Values ')
    plt.ylabel('Predictions ')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
