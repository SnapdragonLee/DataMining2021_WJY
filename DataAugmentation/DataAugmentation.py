from nlpcda import Simbert

import path


def demo():
    config = {
        'model_path': path.get_pretrain_model_path(2),
        'CUDA_VISIBLE_DEVICES': '',
        'max_len': 200,
        'seed': 1
    }
    simbert = Simbert(config=config)
    sent = '把我的一个亿存银行安全吗'
    synonyms = simbert.replace(sent=sent, create_num=5)
    print(synonyms)


if __name__ == '__main__':
    demo()
