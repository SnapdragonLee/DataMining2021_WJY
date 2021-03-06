# from nlpcda import Simbert
import os

from nlpcda import Similarword, RandomDeleteChar
from tqdm import tqdm
from googletrans import Translator
import BERT_Random_Forest
import path
from BERT_Random_Forest import input, const_val


def link_content_and_save(link_num, is_train=True):
    const_val.set_MAX_LINK_NUM_OF_CONTENT(link_num)
    data_set = input.get_data(is_train_data=is_train)
    new_contents = data_set['link_content']
    if is_train:
        with open(path.get_dataset_path('train_data_augment_link{0}.txt'.format(link_num)), 'w',
                  encoding='utf-8') as tar:
            tar.write("id\tcontent\tcharacter\temotions\n")
            for j in tqdm(range(len(new_contents)), desc='train data augment link {0}'.format(link_num)):
                tar.write(
                    "{0}\t{1}\t{2}\t{3}\n".format(data_set['OId'][j], new_contents[j], data_set['characters'][j],
                                                  str(data_set['emotions'][j])[1:-1]))
    else:
        with open(path.get_dataset_path('test_data_link{0}.txt'.format(link_num)), 'w',
                  encoding='utf-8') as tar:
            tar.write("id\tcontent\tcharacter\n")
            for j in tqdm(range(len(new_contents)), desc='test data link {0}'.format(link_num)):
                tar.write(
                    "{0}\t{1}\t{2}\n".format(data_set['OId'][j], new_contents[j], data_set['characters'][j]))


def augment_data():
    NEW_CONTENT_NUM = 2
    data_set = input.get_data(
        data_path=path.get_dataset_path("train_data_augment_link{0}.txt".format(const_val.MAX_LINK_NUM_OF_CONTENT)))

    contents = data_set['contents']
    new_contents = [[] for _ in range(NEW_CONTENT_NUM)]
    for content in tqdm(contents):
        single_new_contents = similar_word(content, NEW_CONTENT_NUM)

        for i in range(NEW_CONTENT_NUM):
            new_contents[i].append(single_new_contents[i])

    for i in range(NEW_CONTENT_NUM):
        with open(path.get_dataset_path(
                'train_data_augment_link_{0}_similar_word{1}.txt'.format(const_val.MAX_LINK_NUM_OF_CONTENT, i)), 'w',
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
        ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????184h????????????????????????????????????3?????????????????????????????????????????????
        ????????????????????????????????????????????????????????????????????????????????????????????????????????????
        ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    '''

    synonyms = simbert.replace(sent=sent, create_num=3)  # ??????????????? 3 (????????????5??????????????????????????????????????????????????????)

    temp = [synonyms[0][0], synonyms[1][0]]
    return temp


if __name__ == '__main__':
    link_content_and_save(3, False)
    augment_data()
    link_content_and_save(4, False)
