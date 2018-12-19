import os

dataset = 'nba'
dst_dir = '{}_data'.format(dataset)
filename_prefix = '{}.'.format(dataset)
fields = ['sent', 'entry', 'attribute', 'value',
          'sent_ref', 'entry_ref', 'attribute_ref', 'value_ref']
modes = ['train', 'val', 'test']
mode_to_filemode = {
    'train': 'train',
    'val': 'valid',
    'test': 'test',
}

train_batch_size = 32
eval_batch_size = 32
batch_sizes = {
    'train': train_batch_size,
    'val': eval_batch_size,
    'test': eval_batch_size,
}

datas = {
    mode: {
        'num_epochs': 1,
        'shuffle': False,
        'batch_size': batch_sizes[mode],
        'allow_smaller_final_batch': True,
        'datasets': [
            {
                'files': [os.path.join(dst_dir, '{}{}.{}.txt'.format(
                    filename_prefix, field, mode_to_filemode[mode]))],
                'vocab_file': os.path.join(dst_dir, '{}.all.vocab.txt'.format(
                    dataset)),
                'data_name': field,
            } for field in fields]
    }
    for mode in modes
}
