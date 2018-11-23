"""Configurations of Transformer model
"""
import copy

n_layers = 6
n_heads = 8
d_k = 64
d_v = 64
d_model = 512
d_inner = d_model * 4
dropout = 0.1

assert d_k == d_v

embedder = {
    'name': 'lookup_table',
    'dim': d_model,
    'initializer': {
        'type': 'random_normal_initializer',
        'kwargs': {
            'mean': 0.0,
            'stddev': d_model**-0.5,
        },
    }
}

encoder = {
    'num_blocks': n_layers,
    'dim': d_model,
    'embedding_dropout': dropout,
    'residual_dropout': dropout,
    'poswise_feedforward': {
        "layers": [
            {
                "type": "Dense",
                "kwargs": {
                    "name": "conv1",
                    "units": d_inner,
                    "activation": "relu",
                    "use_bias": True,
                }
            },
            {
                "type": "Dropout",
                "kwargs": {
                    "rate": 0.1,
                }
            },
            {
                "type": "Dense",
                "kwargs": {
                    "name": "conv2",
                    "units": d_model,
                    "use_bias": True,
                }
            }
        ],
        "name": "ffn"
    },
    'multihead_attention': {
        'num_heads': n_heads,
        'output_dim': d_model,
        'num_units': n_heads * d_k,
        'dropout_rate': 0.1,
    },
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
        },
    }
}

decoder = copy.deepcopy(encoder)
