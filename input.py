import pandas


def read_data(file_path, is_train_data=True):
    script_ids = []
    scene_nums = []
    sentence_nums = []
    ids = []
    contents = []
    characters = []
    emotions = []
    index = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if index > 0:
                item = line.replace('\n', '').split('\t')
                if is_train_data:
                    id, content, character, emotion = item[0], item[1], item[2], item[3]
                else:
                    id, content, character = item[0], item[1], item[2]
                script_id, scene_num, sentence_num = id.split('_')[0], id.split('_')[1], id.split('_')[3]
                script_ids.append(script_id)
                scene_nums.append(scene_num)
                sentence_nums.append(sentence_num)
                ids.append(id)
                contents.append(content)
                characters.append(character)
                if is_train_data:
                    emotions.append(emotion)
            index += 1
    m_ids = {'script_ids': script_ids, 'scene_nums': scene_nums, 'sentence_nums': sentence_nums, 'ids': ids}
    if is_train_data:
        return m_ids, content, characters, emotions
    else:
        return m_ids, content, characters


if __name__ == '__main__':
    data_path = "D:/ML/transformer/test_dataset.tsv"
    data = read_data(data_path, False)
    print(len(data[0]['ids']))
