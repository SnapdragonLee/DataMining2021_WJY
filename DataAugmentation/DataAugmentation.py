# from nlpcda import Simbert
import os

from nlpcda import Similarword, RandomDeleteChar
from tqdm import tqdm
from googletrans import Translator
import BERT_Random_Forest
import path
from BERT_Random_Forest import input


def augment_data():
    NEW_CONTENT_NUM = 2
    data_set = input.get_data()
    contents = data_set['contents']
    new_contents = [[] for _ in range(NEW_CONTENT_NUM)]
    """config = {
        'model_path': path.get_pretrain_model_path(2),
        'CUDA_VISIBLE_DEVICES': 'cuda',
        'max_len': 200,
        'seed': 1
    }
    simbert = Simbert(config=config)"""
    for content in tqdm(contents):
        single_new_contents = del_word(content, NEW_CONTENT_NUM)

        for i in range(NEW_CONTENT_NUM):
            new_contents[i].append(single_new_contents[i])

    for i in range(NEW_CONTENT_NUM):
        with open(path.get_dataset_path('train_data_augment_similar_word{0}.txt'.format(i)), 'w',
                  encoding='utf-8') as tar:
            tar.write("id\tcontent\tcharacter\temotions\n")
            for j in tqdm(range(len(contents)), desc='train data augment {0}'.format(i)):
                tar.write(
                    "{0}\t{1}\t{2}\t{3}\n".format(data_set['OId'][j], new_contents[i][j], data_set['characters'][j],
                                                  str(data_set['emotions'][j])[1:-1]))


smw = Similarword(create_num=5, change_rate=0.4)


def similar_word(origin: str, nums: int):
    rs1 = smw.replace(origin)

    ans = []
    for rs in rs1:
        if rs != origin:
            ans.append(rs)
    if len(ans) == 0:
        ans = rs1
    if len(ans) < nums:
        ans = ans * 5
    return ans[:nums]


rdc = RandomDeleteChar(create_num=5, change_rate=0.3)


def del_word(origin: str, nums: int):
    rs1 = rdc.replace(origin)

    ans = []
    for rs in rs1:
        if rs != origin:
            ans.append(rs)
    if len(ans) == 0:
        ans = rs1
    if len(ans) < nums:
        ans = ans * 5
    return ans[:nums]


def google_trans(origin: str, nums: int):
    trans = Translator()
    t_from = 'zh-cn'
    t_to = 'en'
    times = 0
    ans = []
    while True:
        s = trans.translate(origin, t_to, t_from)
        s = trans.translate(s, t_from, t_to)
        if s != origin:
            print(s)
            print(origin)
            ans.append(s)
            break
        else:
            times += 1
    return ans


def build_sentence(origin: str, nums: int, simbert):
    sent = origin

    '''
        如果要使用这个步骤，那么需要用足够大的内存和足够的算例，假如内存无限大，我需要184h左右的时间算完所有相似据3生成，这是一个相当大的时间开销
        因此不建议使用这个，这里我先放一个暂时可以跑的，也没准自己的代码有问题。
        但是这么看起来没有选择的情况下时间开销已经很大，所以看起来也没什么可以改的，很无奈，而且当句子词汇过大的时候，生成的同义句子会有很大的语义损失
    '''

    synonyms = simbert.replace(sent=sent, create_num=3)  # 暂且定义为 3 (若定义为5，自己的电脑实在无法支撑这么大的运算)

    temp = [synonyms[0][0], synonyms[1][0]]
    return temp


if __name__ == '__main__':
    augment_data()
