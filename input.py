import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import seaborn as env

"""
    数据读取部分，本应做根据序号做key的字典，但目前没有实现
"""


def read_data(file_path, is_train_data=True):
    """
    数据读取
    :param file_path: 文件地址
    :param is_train_data: 是训练数据还是测试数据 默认为训练数据 即含有emotion
    :return: 字典，由 'ids','contents','characters','emotions' 作为key 对于测试数据则没有emotions
    """
    script_ids = []
    scene_nums = []
    sentence_nums = []
    ids = []
    contents = []
    characters = []
    emotions = []
    index = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if index > 0:
                item = line.replace('\n', '').split('\t')
                if is_train_data:
                    id, content, character, emotion = item[0], item[1], item[2], item[3]
                else:
                    id, content, character = item[0], item[1], item[2]
                script_id, scene_num, sentence_num = id.split('_')[0], id.split('_')[1], id.split('_')[3]
                script_ids.append(script_id)
                scene_nums.append(scene_num)
                sentence_nums.append(sentence_num)
                ids.append(id)
                contents.append(content)
                characters.append(character)
                if is_train_data:
                    emotions.append(emotion)
            index += 1
    m_ids = {'script_ids': script_ids, 'scene_nums': scene_nums, 'sentence_nums': sentence_nums, 'ids': ids}
    if is_train_data:
        return {'ids': m_ids, 'contents': contents, 'characters': characters, 'emotions': emotions}
    else:
        return {'ids': m_ids, 'contents': contents, 'characters': characters}


def delete_empty_data(data):
    """
    原始数据中存在情感为空的数据，通过调用该方法，返回一个不含无效句的列表
    :param data: read_data方法的返回值
    :return: 删除无效内容后的返回值
    """
    copy_data = copy.deepcopy(data)
    emotions = copy_data['emotions']
    rm = []
    for i in range(len(emotions)):
        emotion = emotions[i]
        if len(emotion) == 0:
            rm.append(i)
    for key in copy_data:
        ele = copy_data[key]
        if type(ele) is dict:
            for ids_key in ele:
                ids = ele[ids_key]
                assert type(ids) is list
                for i in range(len(rm) - 1, -1, -1):
                    del ids[rm[i]]
        else:
            assert type(ele) is list
            for i in range(len(rm) - 1, -1, -1):
                del ele[rm[i]]

    return copy_data


def split_emotion(emotions):
    """
    情感分解，将read_data()的返回值中emotions对应的值传入，返回将其转化为int列表的形式（原为str）
    :param emotions: read_data()的返回值中emotions对应的值
    :return: emotions 转化为int列表的形式
    """
    new_emotions = [[int(level) for level in emotion_as_str.split(',')]
                    for emotion_as_str in emotions]
    emotions_as_str = []
    for emotion_as_str in emotions:
        s = ''
        l = emotion_as_str.split(',')
        s = s.join(l)
        emotions_as_str.append(s)
    return new_emotions, emotions_as_str


def show_data_distributed(train_data_path):
    """
    观察数据规模的方法，主要包含单个情感程度分布，情感间相关性（注释部分）
    注意：注释部分运行较慢，若想看需等待较长时间
    :param train_data_path 训练数据文件路径
    """

    data = read_data(train_data_path)
    data_with_label = delete_empty_data(data)
    print("before: {0}".format(len(data['contents'])))
    print('after: {0}'.format(len(data_with_label['contents'])))

    data_with_label['emotions'], emotion_as_str = split_emotion(data_with_label['emotions'])

    # plt.figure('diff type emotion hist')
    # plt.hist(emotion_as_str)

    # plt.figure('diff type emotion pie')
    cross_type = Counter(emotion_as_str)
    size = []
    # labels = cross_type.keys()
    for name in cross_type.keys():
        size.append(cross_type[name])
    # plt.pie(size, labels=labels)
    # plt.show()

    print('emotion type num {0}'.format(len(cross_type.keys())))

    emotions = np.array(data_with_label['emotions'])
    panda_f = pd.DataFrame(emotions)
    # env.pairplot(panda_f, kind='kde')
    # plt.show()

    print('exit emotion type and num: {0}'.format(Counter(emotion_as_str)))
    labels = [0, 1, 2, 3]
    for i in range(6):
        plt.figure('emotion{0} his'.format(i))
        plt.hist(emotions[:, i])
        total = len(emotions)
        times = Counter(emotions[:, i])
        print('emotion {0}: {1}'.format(i, times))
        per = []
        num = []
        for j in range(len(times)):
            per.append(times[j] / total)
            num.append(times[j])
        print('emotion {0}: proportion {1}'.format(i, per))
        plt.figure('emotion{0} pie'.format(i))
        plt.pie(num, labels=labels)

    plt.show()


if __name__ == '__main__':
    pass
    test_data_path = "D:/ML/transformer/test_dataset.tsv"
    train_data_path = "D:/ML/transformer/train_dataset_v2.tsv"
    show_data_distributed(train_data_path)
