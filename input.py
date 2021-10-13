import numpy as np
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


def read_data(file_path, is_train_data=True):
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
    new_emotions = [[int(level) for level in emotion_as_str.split(',')]
                    for emotion_as_str in emotions]
    return new_emotions


if __name__ == '__main__':
    test_data_path = "D:/ML/transformer/test_dataset.tsv"
    train_data_path = "D:/ML/transformer/train_dataset_v2.tsv"
    data = read_data(train_data_path)
    data_with_label = delete_empty_data(data)
    print("before: {0}".format(len(data['contents'])))
    print('after: {0}'.format(len(data_with_label['contents'])))
    data_with_label['emotions'] = split_emotion(data_with_label['emotions'])
    emotions = np.array(data_with_label['emotions'])

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
        plt.pie(num)
    plt.show()
