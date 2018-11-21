# Attentional Seq2seq model.
# Hyperparameters not specified here will take the default values.

num_units = 512
embedding_dim = 512
dropout = 0.2

embedder = {
    'dim': embedding_dim
}

encoder = {
    'rnn_cell_fw': {
        'type': 'LSTMCell',
        'kwargs': {
            'num_units': num_units
        },
        'num_layers': 2,
        'dropout': {
            'input_keep_prob': 1. - dropout
        }
    }
}

connector = None

decoder = {
    'rnn_cell': {
        'type': 'LSTMCell',
        'kwargs': {
            'num_units': num_units
        },
        'num_layers': 4,
        'dropout': {
            'input_keep_prob': 1. - dropout
        }
    },
    'attention': {
        'type': 'BahdanauAttention',
        'kwargs': {
            'num_units': num_units,
        },
        'attention_layer_size': num_units
    }
}
