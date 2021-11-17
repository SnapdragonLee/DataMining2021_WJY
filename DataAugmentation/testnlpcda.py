import path
from nlpcda import Simbert


if __name__ == '__main__':
    config = {
        'model_path': path.get_pretrain_model_path(2),
        'CUDA_VISIBLE_DEVICES': '',
        'max_len': 200,
        'seed': 1
    }
    simbert = Simbert(config=config)
    sent = 'd1觉出对方的神色不正常，便怏怏地跑开了。'
    synonyms = simbert.replace(sent=sent, create_num=5)
    print(synonyms)
