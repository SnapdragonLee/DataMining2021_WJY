from transformers import AutoConfig, BertModel, BertTokenizer
from torch import nn
import torch
from bertadjust import init_params
import matplotlib.pyplot as plt

PRE_TRAINED_MODEL_PATH = "path/example"
PRE_TRAINED_MODEL_NAME = 'example'
trained_model_path = "path/example"
test_sentences = ["b2哭的很伤心而a1却哈哈大笑。角色: b2", "b2哭的很伤心而a1却哈哈大笑。角色: a1"]
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_PATH)


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
    _encoding = tokenizer.encode_plus(test_sentences[0],
                                      add_special_tokens=True,
                                      max_length=128,
                                      return_token_type_ids=True,
                                      pad_to_max_length=True,
                                      return_attention_mask=True,
                                      return_tensors='pt', )
    _input_ids = _encoding['input_ids'].flatten(),
    _attention_mask = _encoding['attention_mask'].flatten()
    return _input_ids, _attention_mask


if __name__ == '__main__':
    origin_model = torch.load(trained_model_path)
    params = origin_model.state_dict()
    new_model = IQIYModelLite(1, model_name=PRE_TRAINED_MODEL_NAME, model_path=PRE_TRAINED_MODEL_PATH)
    new_model.load_state_dict(params)
    input_ids, attention_mask = encode(test_sentences[0])
    attention = new_model.get_attention(input_ids=input_ids, attention_mask=attention_mask)
    print(f'text: {test_sentences[0]}')
    print(attention.shape)
    print(new_model(input_ids, attention_mask))
    plt.figure("text1")
    plt.plot(attention)

    input_ids, attention_mask = encode(test_sentences[1])
    attention = new_model.get_attention(input_ids=input_ids, attention_mask=attention_mask)
    print(f'text: {test_sentences[1]}')
    print(attention.shape)
    print(new_model(input_ids, attention_mask))
    plt.figure("text2")
    plt.plot(attention)
    plt.show()
