import os

dataset = "nba"
dst_dir = '{}_data'.format(dataset)
filename_prefix = '{}.'.format(dataset)
fields = ['sent', 'entry', 'attribute', 'value', 'sent_ref', 'entry_ref', 'attribute_ref', 'value_ref']
modes = ['train', 'valid', 'test']

batch_size = 20

data_files = {
    mode: {
        data_name: os.path.join(dst_dir, '{}{}.{}.txt'.format(filename_prefix, data_name, mode))
        for data_name in fields
    }
    for mode in modes
}

datas = {
    stage: {
        "num_epochs": 1,
        "shuffle": stage == 'train',
        "batch_size": batch_size,
        "datasets": [
            {
                "files": [data_files[stage][field]],
                "vocab_file": os.path.join(dst_dir, '{}.all.vocab.txt'.format(dataset)),
                "data_name": field
            } for field in fields]
    }
    for stage in modes
}
