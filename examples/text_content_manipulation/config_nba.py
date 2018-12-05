import os

dataset = "nba"
dst_dir = './%s_data/' % dataset
filename_prefix = '%s.' % dataset
fields = ['sent', 'entry', 'attribute', 'value', 'sent_ref', 'entry_ref', 'attribute_ref', 'value_ref']
modes = ['train', 'test', 'valid']

batch_size = 20
max_num_steps = 20
structured_emb_size = 10
hidden_size = structured_emb_size * 3

data_files = {
    mode: {
        data_name: dst_dir + filename_prefix + '%s.%s.txt' % (data_name, mode)
        for data_name in fields
    }
    for mode in modes
}

data_hparams = {
    stage: {
        "num_epochs": 1,
        "shuffle": stage != 'test',
        "batch_size": batch_size,
        "datasets": [
            {
                "files": [data_files[stage][field]],
                "vocab_file": os.path.join(dst_dir, '%s.all.vocab.txt' % dataset),
                "data_name": field
            } for field in fields]
    }
    for stage in modes
}

emb_hparams = {
    'name': 'lookup_table',
    "dim": hidden_size,
    'initializer': {
        'type': 'random_normal_initializer',
        'kwargs': {
            'mean': 0.0,
            'stddev': hidden_size**-0.5,
        },
    }
}

structured_emb_hparams = {
    'name': 'lookup_table',
    "dim": structured_emb_size,
    'initializer': {
        'type': 'random_normal_initializer',
        'kwargs': {
            'mean': 0.0,
            'stddev': structured_emb_size**-0.5,
        },
    }
}

encoder_hparams = {
    'rnn_cell_fw': {
        'kwargs': {
            'num_units': hidden_size
        }
    }
}

rnn_cell_hparams = {
    "type": "LSTMBlockCell",
    "kwargs": {
        "num_units": hidden_size,
        "forget_bias": 0.
    },
    "dropout": {"output_keep_prob": 0.8},
    "num_layers": 1
}
