from tqdm import tqdm
import pandas as pd
import os
from functools import partial
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel, AutoConfig
from functools import partial
from transformers import AdamW, get_linear_schedule_with_warmup

import os

# 本人电脑只有一个GPU 不需要该句
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# 加载数据
import path

with open(path.get_dataset_path('train_dataset_v2.tsv'), 'r', encoding='utf-8') as handler:
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

test = pd.read_csv(path.get_dataset_path('test_dataset.tsv'), sep='\t')
submit = pd.read_csv(path.get_dataset_path('submit_example.tsv'), sep='\t')
train = train[train['emotions'] != '']

# 数据处理
train['text'] = train['content'].astype(str) + '角色: ' + train['character'].astype(str)
test['text'] = test['content'].astype(str) + ' 角色: ' + test['character'].astype(str)

train['emotions'] = train['emotions'].apply(lambda x: [int(_i) for _i in x.split(',')])

train[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = train['emotions'].values.tolist()
test[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = [0, 0, 0, 0, 0, 0]

train.to_csv(path.get_dataset_path('train.csv'),
             columns=['id', 'content', 'character', 'text', 'love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
             sep='\t',
             index=False)

test.to_csv(path.get_dataset_path('test.csv'),
            columns=['id', 'content', 'character', 'text', 'love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
            sep='\t',
            index=False)

# 定义dataset
target_cols = ['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']


class RoleDataset(Dataset):
    def __init__(self, tokenizer, max_len, mode='train'):
        super(RoleDataset, self).__init__()
        if mode == 'train':
            self.data = pd.read_csv(path.get_dataset_path('train.csv'), sep='\t')
        else:
            self.data = pd.read_csv(path.get_dataset_path('test.csv'), sep='\t')
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
PRE_TRAINED_MODEL_NAME = path.pretrained_model[1]
PRE_TRAINED_MODEL_PATH = path.get_pretrain_model_path(1)  # 'hfl/chinese-roberta-wwm-ext'
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

        self.attention_love = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self.attention_joy = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self.attention_fright = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self.attention_anger = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.attention_fear = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.attention_sorrow = nn.Sequential(
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

        weights_love = self.attention_love(last_layer_hidden_states)
        context_vector_love = torch.sum(weights_love * last_layer_hidden_states, dim=1)

        weights_joy = self.attention_joy(last_layer_hidden_states)
        context_vector_joy = torch.sum(weights_joy * last_layer_hidden_states, dim=1)

        weights_fright = self.attention_fright(last_layer_hidden_states)
        context_vector_fright = torch.sum(weights_fright * last_layer_hidden_states, dim=1)

        weights_anger = self.attention_anger(last_layer_hidden_states)
        context_vector_anger = torch.sum(weights_anger * last_layer_hidden_states, dim=1)

        weights_fear = self.attention_fear(last_layer_hidden_states)
        context_vector_fear = torch.sum(weights_fear * last_layer_hidden_states, dim=1)

        weights_sorrow = self.attention_sorrow(last_layer_hidden_states)
        context_vector_sorrow = torch.sum(weights_sorrow * last_layer_hidden_states, dim=1)

        love = self.out_love(context_vector_love)
        joy = self.out_joy(context_vector_joy)
        fright = self.out_fright(context_vector_fright)
        anger = self.out_anger(context_vector_anger)
        fear = self.out_fear(context_vector_fear)
        sorrow = self.out_sorrow(context_vector_sorrow)

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

model = IQIYModelLite(n_classes=1, model_name=PRE_TRAINED_MODEL_NAME, model_path=PRE_TRAINED_MODEL_PATH)

model.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # correct_bias=False,
total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warm_up_ratio * total_steps,
    num_training_steps=total_steps
)

criterion = nn.BCEWithLogitsLoss().cuda()


# 模型训练
def do_train(model, date_loader, criterion, optimizer, scheduler, metric=None):
    model.train()
    global_step = 0
    tic_train = time.time()
    log_steps = 100
    for epoch in range(EPOCHS):
        losses = []
        for step, sample in enumerate(train_loader):
            input_ids = sample["input_ids"].cuda()
            attention_mask = sample["attention_mask"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss_love = criterion(outputs['love'], sample['love'].view(-1, 1).cuda())
            loss_joy = criterion(outputs['joy'], sample['joy'].view(-1, 1).cuda())
            loss_fright = criterion(outputs['fright'], sample['fright'].view(-1, 1).cuda())
            loss_anger = criterion(outputs['anger'], sample['anger'].view(-1, 1).cuda())
            loss_fear = criterion(outputs['fear'], sample['fear'].view(-1, 1).cuda())
            loss_sorrow = criterion(outputs['sorrow'], sample['sorrow'].view(-1, 1).cuda())
            loss = loss_love + loss_joy + loss_fright + loss_anger + loss_fear + loss_sorrow

            losses.append(loss.item())

            loss.backward()

            #             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % log_steps == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s, lr: %.10f"
                      % (global_step, epoch, step, np.mean(losses), global_step / (time.time() - tic_train),
                         float(scheduler.get_last_lr()[0])))


do_train(model, train_loader, criterion, optimizer, scheduler)

# 模型预测
from collections import defaultdict

model.eval()

test_pred = defaultdict(list)
for step, batch in tqdm(enumerate(valid_loader)):
    b_input_ids = batch['input_ids'].cuda()
    attention_mask = batch["attention_mask"].cuda()
    with torch.no_grad():
        logists = model(input_ids=b_input_ids, attention_mask=attention_mask)
        for col in target_cols:
            out2 = logists[col].sigmoid().squeeze(1) * 3.0
            test_pred[col].append(out2.cpu().numpy())

    print(test_pred)
    break


def predict(model, test_loader):
    val_loss = 0
    test_pred = defaultdict(list)
    model.eval()
    model.cuda()
    for batch in tqdm(test_loader):
        b_input_ids = batch['input_ids'].cuda()
        attention_mask = batch["attention_mask"].cuda()
        with torch.no_grad():
            logists = model(input_ids=b_input_ids, attention_mask=attention_mask)
            for col in target_cols:
                out2 = logists[col].sigmoid().squeeze(1) * 3.0
                test_pred[col].extend(out2.cpu().numpy().tolist())

    return test_pred


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
    path.build_submit_path('baseline_{} EPOCH2 BATCH_SIZE16 MSE.tsv').format(PRE_TRAINED_MODEL_NAME.split('/')[-1]),
    sep='\t',
    index=False)
sub.head()
