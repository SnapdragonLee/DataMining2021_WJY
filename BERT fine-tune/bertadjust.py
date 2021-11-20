import copy

from tqdm import tqdm
import pandas as pd
import os
from functools import partial
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel, AutoConfig
from functools import partial
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error
import os

# 本人电脑只有一个GPU 不需要该句
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# 加载数据
import path

# K折交叉验证
K = 5
PRE_TRAIN_MODEL_NUM = 1
LINK_NUM = 0


def read_data_and_save_as_new_tsv(origin_name, new_tsv_num: int):
    with open(path.get_dataset_path(origin_name), 'r', encoding='utf-8') as handler:
        lines = handler.read().split('\n')[1:-1]
        # 第一句为标签    最后一句为空行

        data = list()
        for line in tqdm(lines):
            sp = line.split('\t')
            if len(sp) != 4:
                print("Error: ", sp)
                continue
            data.append(sp)

    train = pd.DataFrame(data)
    train.columns = ['id', 'content', 'character', 'emotions']
    train = train[train['emotions'] != '']
    train['text'] = train['content'].astype(str) + '角色: ' + train['character'].astype(str)
    train['emotions'] = train['emotions'].apply(lambda x: [int(_i) for _i in x.split(',')])

    train[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = train['emotions'].values.tolist()
    train.to_csv(path.get_dataset_path('train%d.csv' % new_tsv_num),
                 columns=['id', 'content', 'character', 'text', 'love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
                 sep='\t',
                 index=False)


test = pd.read_csv(path.get_dataset_path(path.test_dataset_names[LINK_NUM]), sep='\t')
submit = pd.read_csv(path.get_dataset_path('submit_example.tsv'), sep='\t')

# 数据处理

test['text'] = test['content'].astype(str) + ' 角色: ' + test['character'].astype(str)

test[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = [0, 0, 0, 0, 0, 0]

test.to_csv(path.get_dataset_path('test{}.csv'.format(LINK_NUM)),
            columns=['id', 'content', 'character', 'text', 'love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
            sep='\t',
            index=False)

# 定义dataset
target_cols = ['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']

train_dataframe = pd.read_csv(path.get_dataset_path('train.csv'), sep='\t')


def read_data_and_split_section(number_of_train_set: int):
    _dataframe = pd.read_csv(path.get_dataset_path('train%d.csv' % number_of_train_set), sep='\t')
    # 乱序
    _dataframe = _dataframe.sample(frac=1, random_state=1)
    _dataframe_section = []
    train_dataframe_section_size = len(_dataframe) // K
    for _ in range(K):
        _dataframe_section.append(
            _dataframe[train_dataframe_section_size * _:train_dataframe_section_size * (_ + 1)])
    return _dataframe_section


train_dataframe_sections = list()
for _, file_name in enumerate(path.train_dataset_names[LINK_NUM]):
    read_data_and_save_as_new_tsv(file_name, _)
    train_dataframe_sections.append(read_data_and_split_section(_))


class RoleDataset(Dataset):
    def __init__(self, tokenizer, max_len, _train_dataframe=train_dataframe, mode='train', test_num=LINK_NUM):
        super(RoleDataset, self).__init__()
        if mode == 'train':
            self.data = _train_dataframe
            # print(_train_dataframe['text'])
        else:
            self.data = pd.read_csv(path.get_dataset_path('test{}.csv'.format(test_num)), sep='\t')
        self.texts = self.data['text'].tolist()
        self.labels = self.data[target_cols].to_dict('records')
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(text,
                                              add_special_tokens=True,
                                              max_length=self.max_len,
                                              return_token_type_ids=True,
                                              pad_to_max_length=True,
                                              return_attention_mask=True,
                                              return_tensors='pt', )

        sample = {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        for label_col in target_cols:
            sample[label_col] = torch.tensor(label[label_col] / 3.0, dtype=torch.float)
        return sample

    def __len__(self):
        return len(self.texts)


# create dataloader
def create_dataloader(dataset, batch_size, mode='train'):
    shuffle = True if mode == 'train' else False

    if mode == 'train':
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


# 加载预训练模型
# roberta
PRE_TRAINED_MODEL_NAME = path.pretrained_model[PRE_TRAIN_MODEL_NUM]
PRE_TRAINED_MODEL_PATH = path.get_pretrain_model_path(PRE_TRAIN_MODEL_NUM)  # 'hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_PATH)
base_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_PATH)  # 加载预训练模型


# model = ppnlp.transformers.BertForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=2)

# 模型构建
def init_params(module_lst):
    for module in module_lst:
        for param in module.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    return


class IQIYModelLite(nn.Module):
    def __init__(self, n_classes, model_name, model_path):
        super(IQIYModelLite, self).__init__()
        config = AutoConfig.from_pretrained(model_path)
        config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})

        self.base = BertModel.from_pretrained(model_path, config=config)

        dim = 1024 if 'large' in model_name else 768

        self.attention = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        # self.attention = AttentionHead(h_size=dim, hidden_dim=512, w_drop=0.0, v_drop=0.0)

        self.out_love = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_joy = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_fright = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_anger = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_fear = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
        self.out_sorrow = nn.Sequential(
            nn.Linear(dim, n_classes)
        )

        init_params([self.out_love, self.out_joy, self.out_fright, self.out_anger,
                     self.out_fear, self.out_sorrow, self.attention])

    def forward(self, input_ids, attention_mask):
        roberta_output = self.base(input_ids=input_ids,
                                   attention_mask=attention_mask)

        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)
        # print(weights.size())
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)
        # context_vector = weights

        love = self.out_love(context_vector)
        joy = self.out_joy(context_vector)
        fright = self.out_fright(context_vector)
        anger = self.out_anger(context_vector)
        fear = self.out_fear(context_vector)
        sorrow = self.out_sorrow(context_vector)

        return {
            'love': love, 'joy': joy, 'fright': fright,
            'anger': anger, 'fear': fear, 'sorrow': sorrow,
        }


# 参数配置
EPOCHS = 2
weight_decay = 0.0
data_path = 'data'
warmup_proportion = 0.0
batch_size = 16
lr = 1e-5
max_len = 128

warm_up_ratio = 0.000

trainset = RoleDataset(tokenizer, max_len, mode='train')
train_loader = create_dataloader(trainset, batch_size, mode='train')
valset = RoleDataset(tokenizer, max_len, mode='test')
valid_loader = create_dataloader(valset, batch_size, mode='test')


def build_model():
    model = IQIYModelLite(n_classes=1, model_name=PRE_TRAINED_MODEL_NAME, model_path=PRE_TRAINED_MODEL_PATH)

    model.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # correct_bias=False,
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warm_up_ratio * total_steps,
        num_training_steps=total_steps
    )
    return model, optimizer, scheduler


total_steps = len(train_loader) * EPOCHS

criterion = nn.BCEWithLogitsLoss().cuda()


def predict(model, test_loader, title=''):
    val_loss = 0
    test_pred = defaultdict(list)
    model.eval()
    model.cuda()
    for batch in tqdm(test_loader, desc='Pred' + title):
        b_input_ids = batch['input_ids'].cuda()

        attention_mask = batch["attention_mask"].cuda()
        with torch.no_grad():
            logists = model(input_ids=b_input_ids, attention_mask=attention_mask)
            for col in target_cols:
                out2 = logists[col].sigmoid().squeeze(1) * 3.0
                test_pred[col].extend(out2.cpu().numpy().tolist())

    return test_pred


# 构建 第Ki折的数据
def get_ki_dataframe(ki: int, data_num: int):
    train_dataframe_section = train_dataframe_sections[data_num]
    k_train_dataframe = pd.DataFrame()
    k_test_dataframe = pd.DataFrame()
    for _ in range(K):
        if _ != ki:
            k_train_dataframe = pd.concat([k_train_dataframe, train_dataframe_section[_]])
        else:
            k_test_dataframe = train_dataframe_section[_]
    return k_train_dataframe, k_test_dataframe


BESTMODEL_NAME = 'K{0} link{1} bestModel{2}.nn'


# 模型训练
def do_train(criterion, metric=None, K=5):
    best_model = None
    best_performance = 1
    global_step = 0
    tic_train = time.time()
    log_steps = 100
    for k in range(K):
        k_model, optimizer, scheduler = build_model()
        k_model.cuda()
        k_model.train()
        k_test_dataframe = pd.DataFrame()

        for epoch in range(EPOCHS):
            losses = []
            k_train_dataframe, k_test_dataframe_section = get_ki_dataframe(k, epoch)
            k_test_dataframe = pd.concat([k_test_dataframe, k_test_dataframe_section])
            k_train_set = RoleDataset(tokenizer, max_len, k_train_dataframe, mode='train')
            k_train_loader = create_dataloader(k_train_set, batch_size, mode='train')
            for step, sample in enumerate(k_train_loader):
                # print(sample)
                input_ids = sample["input_ids"].cuda()
                attention_mask = sample["attention_mask"].cuda()

                outputs = k_model(input_ids=input_ids, attention_mask=attention_mask)

                loss_love = criterion(outputs['love'], sample['love'].view(-1, 1).cuda())
                loss_joy = criterion(outputs['joy'], sample['joy'].view(-1, 1).cuda())
                loss_fright = criterion(outputs['fright'], sample['fright'].view(-1, 1).cuda())
                loss_anger = criterion(outputs['anger'], sample['anger'].view(-1, 1).cuda())
                loss_fear = criterion(outputs['fear'], sample['fear'].view(-1, 1).cuda())
                loss_sorrow = criterion(outputs['sorrow'], sample['sorrow'].view(-1, 1).cuda())
                loss = loss_love + loss_joy + loss_fright + loss_anger + loss_fear + loss_sorrow
                # print(loss)
                losses.append(loss.item())

                loss.backward()

                #             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % log_steps == 0:
                    print(
                        "Cross Validation %d Of %d global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s, lr: %.10f"
                        % (k, K, global_step, epoch, step, np.mean(losses), global_step / (time.time() - tic_train),
                           float(scheduler.get_last_lr()[0])))

        k_test_set = RoleDataset(tokenizer, max_len, k_test_dataframe, mode='train')
        k_test_loader = create_dataloader(k_test_set, batch_size, mode='test')
        k_test_pred = predict(k_model, k_test_loader, title=str(k))
        k_test_pred = np.array(
            [[k_test_pred[col][_] for col in target_cols] for _ in range(len(k_test_set.labels))])
        k_test_label = np.array(
            [[k_test_set.labels[_][col] for col in target_cols] for _ in range(len(k_test_set.labels))])

        k_mse = mean_squared_error(y_true=k_test_label, y_pred=k_test_pred)
        print("K=%d MSE=%.5f RMSE=%.5f" % (k, k_mse, np.sqrt(k_mse)))
        if k_mse < best_performance:
            torch.save(k_model, path.build_model_path(BESTMODEL_NAME).format(K, LINK_NUM, PRE_TRAINED_MODEL_NAME))
            del k_model
            best_performance = k_mse
    if best_performance != 1:
        print('Best Model MSE=%.5f RMSE=%.5f Scores=%.5f' % (
            best_performance, np.sqrt(best_performance), 1 / (1 + np.sqrt(best_performance))))
        best_model = torch.load(
            path.build_model_path(BESTMODEL_NAME).format(K, LINK_NUM, PRE_TRAINED_MODEL_NAME))
    return best_model


"""
##############################################
# 这部分代码为模型软投票
##############################################
def merge_pred(models: list, valid_loaders: list):
    label_preds = np.zeros(6)
    for model_num, model in enumerate(models):
        valid_loader = valid_loaders[model_num]
        test_pred = predict(model, valid_loader)
        single_label_preds = []
        for col in target_cols:
            preds = test_pred[col]
            single_label_preds.append(preds)
        label_preds = label_preds + np.stack(single_label_preds, axis=1)
    label_preds = label_preds / len(models)
    submit = pd.read_csv(path.get_dataset_path('submit_example.tsv'), sep='\t')
    print(len(label_preds[0]))
    sub = submit.copy()
    sub['emotion'] = label_preds.tolist()
    sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(i) for i in x]))
    sub.to_csv(
        path.build_submit_path('MergeAns.tsv'),
        sep='\t',
        index=False)
    sub.head()


models = list()
valid_loaders = list()
for i, model_name in enumerate(path.MODELS_NAME):
    models.append(torch.load(path.build_model_path(model_name)))
    val_set = RoleDataset(tokenizer, max_len, mode='test', test_num=i)
    valid_loader = create_dataloader(val_set, batch_size, mode='test')
    valid_loaders.append(valid_loader)
merge_pred(models, valid_loaders)
##############################################
"""
model = do_train(criterion)

# 模型预测


model.eval()

test_pred = defaultdict(list)
for step, batch in tqdm(enumerate(valid_loader), desc='Final Pred'):
    b_input_ids = batch['input_ids'].cuda()
    attention_mask = batch["attention_mask"].cuda()
    with torch.no_grad():
        logists = model(input_ids=b_input_ids, attention_mask=attention_mask)
        for col in target_cols:
            out2 = logists[col].sigmoid().squeeze(1) * 3.0
            test_pred[col].append(out2.cpu().numpy())

    print(test_pred)
    break

submit = pd.read_csv(path.get_dataset_path('submit_example.tsv'), sep='\t')
test_pred = predict(model, valid_loader)

print(test_pred['love'][:10])
print(len(test_pred['love']))

label_preds = []
for col in target_cols:
    preds = test_pred[col]
    label_preds.append(preds)
print(len(label_preds[0]))
sub = submit.copy()
sub['emotion'] = np.stack(label_preds, axis=1).tolist()
sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(i) for i in x]))
sub.to_csv(
    path.build_submit_path('K test link{0} {1}.tsv').format(LINK_NUM, PRE_TRAINED_MODEL_NAME.split('/')[-1]),
    sep='\t',
    index=False)
sub.head()
