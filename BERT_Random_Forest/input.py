import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import path
from BERT_Random_Forest import const_val as c_v

"""
    pip install seaborn
    pip install pandas
    数据读取部分，本应做根据序号做key的字典，但目前没有实现
"""


def build_train_data():
    """
    构造训练数据 其中包括原语句与扩展语句 需要调用随机语句
    :return: y, x 数据与标签
    """
    dic = get_data()
    y = []
    x = []
    link_content_merged = dic['link_content_merged']
    merged_sentences = dic['merged_sentences']
    emotions = dic['emotions']
    for i in range(len(emotions)):
        # x.append(link_content_merged[i])
        # y.append(emotions[i])
        x.append(merged_sentences[i])
        y.append(emotions[i])
    print('build data')

    classifier_y = get_classifier_data(y)
    encoder = LabelEncoder()
    encoder = encoder.fit(classifier_y)
    classifier_y = encoder.transform(classifier_y)
    return y, x, classifier_y, encoder


def get_data(is_train_data=True, data_path=None):
    """
    读取数据的函数，
    :return: dic由 'ids','contents','characters','emotions' ,'merged_sentences'作为key
            调用时可用 merged_sentences中的句子 句子形式 content + '[MASK]角色' + character
            新增 key ‘link_content’ 向前拼接的句子 'link_content_merged' 句子增加角色名
    """
    data_path = data_path if data_path is not None \
        else path.get_origin_train_data_path() if is_train_data else \
        path.get_origin_train_data_path(False)
    data, table = read_data(data_path, is_train_data)
    if is_train_data:
        data = delete_empty_data(data)
        data['emotions'] = split_emotion(data['emotions'])
    linked_content = sentence_merging_up(data, table)
    data['link_content'] = linked_content
    merged_sentences_link = sentence_merging_character(data, 'link_content')
    merged_sentences = sentence_merging_character(data)
    data['merged_sentences'] = merged_sentences
    data['link_content_merged'] = merged_sentences_link
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
    origin_ids = []
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
                    id = (int(script_id), int(scene_num), int(sentence_num))
                origin_ids.append(origin_id)
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
        return {'ids': ids, 'contents': contents, 'characters': characters, 'emotions': emotions,
                'OId': origin_ids}, content_dic
    else:
        return {'ids': ids, 'contents': contents, 'characters': characters, 'OId': origin_ids}, content_dic


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


def get_classifier_data(emotions):
    emotions = np.array(emotions)
    need_emotion_combination = (100, 0, 1, 1000, 10, 101, 1100)
    # Counter({0: 20526, 100: 7760, 1: 3062, 1000: 2113, 10: 1463, 101: 554, 1100: 475,
    # 1010: 276, 110: 270, 11: 174, 1001: 70, 1110: 15, 111: 12, 1101: 8, 1111: 2, 1011: 2})
    love = emotions[:, 0]
    joy = emotions[:, 1]
    shock = emotions[:, 2]
    anger = emotions[:, 3]
    fear = emotions[:, 4]
    sad = emotions[:, 5]
    pn_data = []
    for i in range(len(emotions)):
        ans = 0
        level = []
        if love[i] > 0 or joy[i] > 0:
            ans += 1
            level.append(love[i] + joy[i])
        if shock[i] > 0:
            ans += 10
            level.append(shock[i])
        if anger[i] > 0 or sad[i] > 0:
            ans += 100
            level.append(anger[i] + sad[i])
        if fear[i] > 0:
            ans += 1000
            level.append(fear[i])
        if ans not in need_emotion_combination:
            ans = 10 ** level.index(max(level))
        pn_data.append(ans)
    return pn_data


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
            ans += 10
        if anger[i] > 0 or sad[i] > 0:
            ans += 100
        if fear[i] > 0:
            ans += 1000
        pn_data.append(ans)
    counter = Counter(pn_data)
    print(counter)
    plt.hist(pn_data)
    plt.show()


def sentence_merging_character(_data: dict, origin_key='contents'):
    sentences = _data[origin_key]
    characters = _data['characters']
    merged_sentences = []
    for i in range(len(sentences)):
        merged_sentence = sentences[i] + '[MASK]角色：' + characters[i]
        merged_sentences.append(merged_sentence)
    return merged_sentences


def sentence_merging_up(basic_data_set: dict, id2content_data_set: dict, max_link_size=c_v.MAX_SIZE_OF_CONTENT):
    # 最大拼接长度
    maxsize = max_link_size
    # 拼接句子数量
    link_len = c_v.MAX_LINK_NUM_OF_CONTENT

    link_sent = []
    sent_ids = basic_data_set['ids']

    for i in range(0, len(sent_ids)):
        linked_num = 0
        sent_size = 0
        id = sent_ids[i]
        link_sent.append(id2content_data_set[id])
        for j in range(1, i - 1):
            look_for = (id[0], id[1], id[2] - j)
            look_for_s = id2content_data_set.get(look_for)
            look_for_same = (id[0], id[1], id[2] - j + 1)
            look_for_same_s = id2content_data_set.get(look_for_same)
            # 判断是否相同
            if look_for_s is None:
                break
            if look_for_s == look_for_same_s:
                continue

            sent_size += len(look_for_s)

            # 大于规定大小则停止
            if sent_size >= maxsize:
                break

            link_sent[i] = look_for_s + link_sent[i]
            linked_num += 1
            # 达到要求拼接数量则停止
            if linked_num == link_len:
                break
    return link_sent


def main():
    data_set = get_data()
    show_positive_negative_mix(data_set)


if __name__ == '__main__':
    print(c_v.MAX_SIZE_OF_CONTENT)
