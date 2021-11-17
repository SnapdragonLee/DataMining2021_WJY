# from nlpcda import Simbert
from tqdm import tqdm

import BERT_Random_Forest
import path
from BERT_Random_Forest import input


def augment_data():
    NEW_CONTENT_NUM = 2
    data_set = input.get_data()
    contents = data_set['contents']
    new_contents = [[]] * NEW_CONTENT_NUM
    for content in tqdm(contents):
        single_new_contents = build_sentence(content, NEW_CONTENT_NUM)
        for i in range(NEW_CONTENT_NUM):
            new_contents[i].append(single_new_contents[i])
    for i in range(NEW_CONTENT_NUM):
        with open(path.get_dataset_path('train_data_augment{0}.txt'.format(i)), 'w') as tar:
            for j in tqdm(range(len(contents)), desc='train data augment{0}'.format(i)):
                tar.write(
                    "{0}\t{1}\t{2}\t{3}\n".format(data_set['OId'][j], new_contents[i][j], data_set['characters'][j],
                                                  str(data_set['emotions'][j]).[1, -1]))


def build_sentence(origin: str, nums: int):
    pass
    return ['test'] * 2


"""def demo():
    config = {
        'model_path': path.get_pretrain_model_path(2),
        'CUDA_VISIBLE_DEVICES': '',
        'max_len': 200,
        'seed': 1
    }
    simbert = Simbert(config=config)
    sent = '把我的一个亿存银行安全吗'
    synonyms = simbert.replace(sent=sent, create_num=5)
    print(synonyms)"""

if __name__ == '__main__':
    augment_data()
