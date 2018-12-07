structured_emb_size = 128
hidden_size = structured_emb_size * 3

sent_embedder = {
    'name': 'sent_embedder',
    "dim": hidden_size,
    'initializer': {
        'type': 'random_normal_initializer',
        'kwargs': {
            'mean': 0.0,
            'stddev': hidden_size**-0.5,
        },
    }
}

sd_embedder = {
    'name': 'sd_embedder',
    "dim": structured_emb_size,
    'initializer': {
        'type': 'random_normal_initializer',
        'kwargs': {
            'mean': 0.0,
            'stddev': structured_emb_size**-0.5,
        },
    }
}

sent_encoder = {
    'name': 'sent_encoder',
    'rnn_cell_fw': {
        'kwargs': {
            'num_units': hidden_size
        }
    }
}

sd_encoder = {
    'name': 'sd_encoder',
    'rnn_cell_fw': {
        'kwargs': {
            'num_units': hidden_size
        }
    }
}

rnn_cell = {
    "type": "LSTMBlockCell",
    "kwargs": {
        "num_units": hidden_size,
        "forget_bias": 0.
    },
    "dropout": {"output_keep_prob": 0.8},
    "num_layers": 1
}

decoder = {
    "name": "decoder"
}
