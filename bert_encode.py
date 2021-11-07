import torch
from transformers import *
import transformers as tfs
import os
import tensorflow
import numpy as np

import path

"""
    pip install tensorflow
    pip install transformers
    install pytorch in it website by yourself
"""


class BertEncoder:
    model: BertModel
    tokenizer_class: BertTokenizer

    def __init__(self, pretrained_model_path):
        """
        初始化 读入模型与分词器
        :param pretrained_model_path: 模型文件（文件夹名称）
        """
        self.model = BertModel.from_pretrained(pretrained_model_path, from_tf=False)
        self.tokenizer_class = BertTokenizer.from_pretrained(pretrained_model_path)
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model.cpu()

    def chinese2encode_bert(self, sentences: list):
        """
        通过预训练模型得到预编码后的特征向量组
        :param sentences: 语句序列
        :return: 特征向量组
        """
        # assert type(sentences) is tuple
        split_batch = 100

        train_tokenized = [self.tokenizer_class.encode(text) for text in sentences]
        train_max_len = 0
        for i in train_tokenized:
            if len(i) > train_max_len:
                train_max_len = len(i)

        padded = np.array([i + [0] * (train_max_len - len(i)) for i in train_tokenized])


        features = np.empty(shape=[0, 768])
        for i in range(padded.shape[0] // split_batch + 1):
            print("finish{0}/{1}".format(i, padded.shape[0] // split_batch + 1))
            padded_batch = np.array(padded[i * split_batch:i * split_batch + split_batch])
            attention_mask = np.where(padded_batch != 0, 1, 0)
            """print(padded_batch)"""
            input_ids = torch.tensor(padded_batch)
            input_ids = input_ids.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            attention_mask = torch.tensor(attention_mask)
            attention_mask = attention_mask.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


            with torch.no_grad():
                last_hidden_states = self.model(input_ids, attention_mask=attention_mask)

            temp = last_hidden_states[0][:, 0, :].cpu().numpy()

            """
            print(i)
            print(temp.shape)
            print(features.shape)
            print(features)
            """
            features = np.append(features, temp, axis=0)


        print('ans shape:' + str(features.shape))
        # print(features)

        return features

    def stitch_characters_and_encode(self, data_dic: dict):
        y = data_dic['emotions']
        x = self.chinese2encode_bert(data_dic['merged_sentences'])
        return y, x


def main():
    pretrained_model = path.pretrained_model
    st = ["这是一个测试语句", "这是一个苹果", "我爱吃苹果", "这是一个测试语句", "这是一个苹果", "我爱吃苹果"]
    bert_encoder1 = BertEncoder(pretrained_model)
    bert_encoder1.chinese2encode_bert(st)


if __name__ == '__main__':
    main()
