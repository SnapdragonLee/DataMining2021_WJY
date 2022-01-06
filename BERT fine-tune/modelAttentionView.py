from transformers import AutoConfig, BertModel, BertTokenizer
from torch import nn
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

PRE_TRAINED_MODEL_PATH = "..\\chinese-macbert-large"
PRE_TRAINED_MODEL_NAME = 'chinese-macbert-large'
trained_model_path = ".\\Model\\K5 link0 bestModelchinese-macbert-large.nn"
test_sentences = ["b2哭的很伤心而a1却哈哈大笑。角色: b2", "b2哭的很伤心而a1却哈哈大笑。角色: a1"]
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_PATH)

target_cols = ['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']


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

    def get_attention(self, input_ids, attention_mask):
        roberta_output = self.base(input_ids=input_ids,
                                   attention_mask=attention_mask)

        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)
        return weights


def encode(text):
    _encoding = tokenizer.encode_plus(text,
                                      add_special_tokens=True,
                                      max_length=25,
                                      return_token_type_ids=True,
                                      pad_to_max_length=True,
                                      return_attention_mask=True,
                                      return_tensors='pt', )

    _input_ids = torch.unsqueeze(_encoding['input_ids'].flatten(), 0)

    _attention_mask = torch.unsqueeze(_encoding['attention_mask'].flatten(), 0)

    return _input_ids, _attention_mask


def show_pred(logists):
    ans = defaultdict(list)
    for col in target_cols:
        out2 = logists[col].sigmoid().squeeze(1) * 3.0
        ans[col].append(out2.cpu().numpy())
    return ans


if __name__ == '__main__':
    origin_model = torch.load(trained_model_path)
    params = origin_model.state_dict()
    new_model = IQIYModelLite(1, model_name=PRE_TRAINED_MODEL_NAME, model_path=PRE_TRAINED_MODEL_PATH)
    new_model.load_state_dict(params)
    new_model.eval()
    with torch.no_grad():
        input_ids, attention_mask = encode(test_sentences[0])
        attention = new_model.get_attention(input_ids=input_ids, attention_mask=attention_mask)
        print(f'text: {test_sentences[0]}')
        print(attention.shape)
        attention = attention.numpy()
        attention = attention.flatten()
        print(show_pred(new_model(input_ids, attention_mask)))
        plt.figure("text1")
        plt.plot(attention)

        input_ids, attention_mask = encode(test_sentences[1])
        attention = new_model.get_attention(input_ids=input_ids, attention_mask=attention_mask)
        print(f'text: {test_sentences[1]}')
        print(attention.shape)
        attention = attention.numpy()
        attention = attention.flatten()
        print(show_pred(new_model(input_ids, attention_mask)))
        plt.figure("text2")
        plt.plot(attention)
        plt.show()
