import torch
from transformers import *
import transformers as tfs
import os
import tensorflow
import numpy as np

"""
    pip install tensorflow
    pip install transformers
    install pytorch in it website by yourself
"""


class bert_encoder:
    model: BeitModel
    tokenizer_class: BertTokenizer

    def __init__(self, pretrained_model_path):
        """
        初始化 读入模型与分词器
        :param pretrained_model_path: 模型文件（文件夹名称）
        """
        self.model = BertModel.from_pretrained(pretrained_model_path, from_tf=True)
        self.tokenizer_class = BertTokenizer.from_pretrained(pretrained_model_path)

    def chinese2encode_bert(self, sentences):
        """
        通过预训练模型得到预编码后的特征向量组
        :param sentences: 语句序列 应为元组
        :return: 特征向量组
        """
        assert type(sentences) is tuple

        train_tokenized = [self.tokenizer_class.encode(test) for test in sentences]
        print(train_tokenized)
        train_max_len = 0
        for i in train_tokenized:
            if len(i) > train_max_len:
                train_max_len = len(i)

        padded = np.array([i + [0] * (train_max_len - len(i)) for i in train_tokenized])
        print("train set shape:", padded.shape)

        attention_mask = np.where(padded != 0, 1, 0)

        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)

        with torch.no_grad():
            last_hidden_states = self.model(input_ids, attention_mask=attention_mask)

        features = last_hidden_states[0][:, 0, :].numpy()

        print(features.shape)
        print(features)


if __name__ == '__main__':
    pretrained_model = "D:/ML/transformer/bert-base-chinese/"
    st = ("这是一个测试语句", '这也是一个测试语句')
    bert_encoder1 = bert_encoder(pretrained_model)
    bert_encoder1.chinese2encode_bert(st)
