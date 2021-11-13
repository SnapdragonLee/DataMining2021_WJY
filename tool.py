import matplotlib.pyplot as plt


def liner_plot(ans, predictions, figure_name):
    plt.figure('emotion id ' + figure_name)
    plt.scatter(ans, predictions)
    plt.xlabel('True Values ')
    plt.ylabel('Predictions ')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])

    diff = [[], [], [], []]
    for i in range(len(ans)):
        diff[predictions[i]].append(abs(ans[i] - predictions[i]))
    for i in range(4):
        plt.figure('emotion id ' + figure_name + 'level ' + str(i))
        plt.hist(diff[i])


def main():
    ans = [1, 2, 3, 4, 5]
    pre = [0, 1, 2, 1, 3]
    n = '1'
    liner_plot(ans, pre, n)
    plt.show()


if __name__ == '__main__':
    main()
