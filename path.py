import os

pretrained_model = (
    "bert-base-chinese", "chinese-roberta-wwm-ext", "chinese-macbert-large", "chinese-roberta-wwm-ext-large",
    "chinese_simbert_L-12_H-768_A-12")
origin_test_data_name = "test_dataset.tsv"
origin_train_data_name = "train_dataset_v2.tsv"
dataset_names = [origin_train_data_name, 'train_data_augment_similar_word0.txt', "train_data_augment_del_word0.txt",
                 'train_data_augment_similar_word1.txt', "train_data_augment_del_word1.txt"]


def build_submit_path(name: str):
    return os.path.join(os.path.abspath('..'), 'submit file', name)


def build_tmp_data_path(name: str):
    return os.path.join(os.path.abspath('..'), 'RF tmpData', name)


def build_random_forest_model_path(name: str):
    return os.path.join(os.path.abspath('..'), 'RandomForestModel', name)


def build_model_path(name: str):
    return os.path.join(os.path.abspath('..'), 'Model', name)


def get_origin_train_data_path(is_train_data=True):
    return os.path.join(os.path.abspath('..\\..'), "dataset",
                        origin_train_data_name if is_train_data else origin_test_data_name)


def get_dataset_path(name: str):
    return os.path.join(os.path.abspath('..\\..'), "dataset", name)


def get_pretrain_model_path(index: int):
    return os.path.join(os.path.abspath('..\\..'), pretrained_model[index])


if __name__ == '__main__':
    print(os.path.abspath('..'))
