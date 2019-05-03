import os

dataset_dir = 'nba_data'
mode_to_filemode = {
    'train': 'train',
    'val': 'valid',
    'test': 'test',
}
modes = list(mode_to_filemode.keys())
field_to_vocabname = {
    'x_value': 'x_value',
    'x_type': 'x_type',
    'x_associated': 'x_associated',
    'y_aux': 'y',
    'x_ref_value': 'x_value',
    'x_ref_type': 'x_type',
    'x_ref_associated': 'x_associated',
    'y_ref': 'y',
}
fields = list(field_to_vocabname.keys())

train_batch_size = 32
eval_batch_size = 32

datas = {
    mode: {
        'num_epochs': 1,
        'shuffle': mode == 'train',
        'batch_size': train_batch_size if mode == 'train' else eval_batch_size,
        'allow_smaller_final_batch': mode != 'train',
        'datasets': [
            {
                'files': [
                    os.path.join(
                        dataset_dir, mode,
                        '{}.{}.txt'.format(field, mode_to_filemode[mode])
                    )
                ],
                'vocab_file': os.path.join(
                    dataset_dir,
                    '{}.vocab.txt'.format(field_to_vocabname[field])),
                'data_name': field,
            }
            for field in fields
        ]
    }
    for mode in modes
}
