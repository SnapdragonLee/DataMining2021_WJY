import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import seaborn as env

import path

"""
    pip install seaborn
    pip install pandas
    数据读取部分，本应做根据序号做key的字典，但目前没有实现
"""


def get_data(is_train_data=True):
    """
    读取数据的函数，
    :return: dic由 'ids','contents','characters','emotions' ,'merged_sentences'作为key
            调用时可用 merged_sentences中的句子 句子形式 content + '[MASK]角色' + character
    """
    data_path = path.train_data_path if is_train_data else path.test_data_path
    data, table = read_data(data_path, is_train_data)
    if is_train_data:
        data = delete_empty_data(data)
        data['emotions'] = split_emotion(data['emotions'])
    data = sentence_merging_character(data)
    return data


def read_data(file_path, is_train_data=True):
    """
    数据读取
    :param file_path: 文件地址
    :param is_train_data: 是训练数据还是测试数据 默认为训练数据 即含有emotion
    :return:    字典1，由 'ids','contents','characters','emotions' 作为key 对于测试数据则没有emotions
                字典2，id对应语句内容，便于查找前后文
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
                origin_id = id
                script_id, scene_num, sentence_num = id.split('_')[0], id.split('_')[1], id.split('_')[3]
                if is_train_data:
                    id = (int(script_id), int(scene_num), int(sentence_num))
                else:
                    id = (origin_id, int(script_id), int(scene_num), int(sentence_num))
                script_ids.append(script_id)
                scene_nums.append(scene_num)
                sentence_nums.append(sentence_num)
                ids.append(id)
                contents.append(content)
                characters.append(character)
                if is_train_data:
                    emotions.append(emotion)
            index += 1
    content_dic = {}
    for i in range(len(ids)):
        content_dic[ids[i]] = contents[i]

    if is_train_data:
        return {'ids': ids, 'contents': contents, 'characters': characters, 'emotions': emotions}, content_dic
    else:
        return {'ids': ids, 'contents': contents, 'characters': characters}, content_dic


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


def split_emotion(emotions, get_emotion_as_str=False):
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
    if get_emotion_as_str:
        return new_emotions, emotions_as_str
    else:
        return new_emotions


def show_data_distributed(data_path):
    """
    观察数据规模的方法，主要包含单个情感程度分布，情感间相关性（注释部分）
    注意：注释部分运行较慢，若想看需等待较长时间
    :param data_path 训练数据文件路径
    """

    data, content_dic = read_data(data_path)
    data_with_label = delete_empty_data(data)
    print("before: {0}".format(len(data['contents'])))
    print('after: {0}'.format(len(data_with_label['contents'])))

    data_with_label['emotions'], emotion_as_str = split_emotion(data_with_label['emotions'], True)

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


def show_positive_negative_mix(data_set: dict):
    emotions1 = data_set['emotions']
    emotions = np.array(emotions1)
    love = emotions[:, 0]
    joy = emotions[:, 1]
    shock = emotions[:, 2]
    anger = emotions[:, 3]
    fear = emotions[:, 4]
    sad = emotions[:, 5]

    """
    pd_f = pd.DataFrame(emotions1, columns=['love', 'joy', 'shock', 'anger', 'fear', 'sad'])
    env.pairplot(pd_f, kind='kde')
    plt.show()
    """
    pn_data = []
    for i in range(len(emotions)):
        ans = 0
        if love[i] > 0 or joy[i] > 0:
            ans += 1
        if shock[i] > 0:
            ans += 2
        if anger[i] > 0 or sad[i] > 0:
            ans += 4
        if fear[i] > 0:
            ans += 16
        pn_data.append(ans)
    counter = Counter(pn_data)
    print(counter)
    plt.hist(pn_data)
    plt.show()


def sentence_merging_character(_data: dict):
    sentences = _data['contents']
    characters = _data['characters']
    merged_sentences = []
    for i in range(len(sentences)):
        merged_sentence = sentences[i] + '[MASK]角色：' + characters[i]
        merged_sentences.append(merged_sentence)
    _data['merged_sentences'] = merged_sentences;
    return _data


def sentence_merging_up(basic_data_set: dict, id2content_data_set: dict):
    #最大拼接长度
    maxsize = 400
    #拼接句子数量
    link_len = 0
    linked_num= 0
    link_sent = []
    sent_ids = basic_data_set['ids']
    sent_id = id2content_data_set['id']
    sentence = id2content_data_set['contents']
    for i in range(0, len(sentence)):
        sent_size = 0
        for j in range(i, 0):
            #判断是否相同
            if sentence[j] == sentence[j - 1]:
                continue
            #id前两项相等则拼接
            if sent_ids[i][1] == sent_id[j][1] and sent_ids[i][2] == sent_id[j][2]:
                sent_size += sentence[j].size()
                #大于规定大小则停止
                if sent_size >= maxsize:
                    break;
                link_sent[i] = sentence[j] + link_sent[i];
                linked_num += 1;
            #达到要求拼接数量则停止
            if linked_num == link_len:
                break
    return link_sent


def main():
    test_data_path = path.test_data_path
    train_data_path = path.train_data_path
    data, table = read_data(train_data_path)
    data = delete_empty_data(data)
    data['emotions'] = split_emotion(data['emotions'])
    print('pretreated')
    show_positive_negative_mix(data)
    data = sentence_merging_character(data)
    # print(data['merged_sentences'])


if __name__ == '__main__':
    pass
    main()
