import os

pretrained_model = ("bert-base-chinese", "chinese-roberta-wwm-ext")
origin_test_data_name = "test_dataset.tsv"
origin_train_data_name = "train_dataset_v2.tsv"


def build_submit_path(name: str):
    return os.path.join(os.path.abspath('./'), 'submit file', name)


def build_tmp_data_path(name: str):
    return os.path.join(os.path.abspath('./'), 'RF tmpData', name)


def build_random_forest_model_path(name: str):
    return os.path.join(os.path.abspath('./'), 'RandomForestModel', name)


def get_origin_train_data_path(is_train_data=True):
    return os.path.join(os.path.abspath('../'), "dataset",
                        origin_train_data_name if is_train_data else origin_test_data_name)


def get_dataset_path(name: str):
    return os.path.join(os.path.abspath('../'), "dataset", name)


def get_pretrain_model_path(index: int):
    return os.path.join(os.path.abspath('../'), pretrained_model[index])
