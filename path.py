pretrained_model = "../bert-base-chinese"
test_data_path = "../test_dataset.tsv"
train_data_path = "../train_dataset_v2.tsv"


def build_submit_path(name: str):
    return './submit file/' + name


def build_tmp_data_path(name: str):
    return './tmpData/' + name


def build_model_path(name: str):
    return './regressorModel/' + name
