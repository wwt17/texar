"""Configurations of Transformer model
"""
import copy
import texar as tx

random_seed = 0
beam_width = 1
alpha = 0.6
n_layers = 8
n_heads = 5
d_k = 64
d_v = 64
d_model = 288
d_inner = 507
init = 0.035
dropout = 0.25

assert d_k == d_v

initializer = {
    'type': 'random_uniform_initializer',
    'kwargs': {
        'minval': -init,
        'maxval':  init,
    }
}

emb = {
    'name': 'lookup_table',
    'dim': d_model,
    'initializer': initializer
}

encoder = {
    'num_blocks': n_layers,
    'dim': d_model,
    'embedding_dropout': 1 - dropout,
    'residual_dropout': 1 - dropout,
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
    'initializer': initializer
}

decoder = copy.deepcopy(encoder)

loss_label_confidence = 0.9

opt = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'beta1': 0.9,
            'beta2': 0.997,
            'epsilon': 1e-9,
        }
    },
    'learning_rate_decay': {
        'type': 'exponential_decay',
        'kwargs': {
            'learning_rate': 1e-3,
            'decay_steps': 1000,
            'decay_rate': 0.97,
        },
        'start_decay_step': 8000
    },
    'gradient_clip': {
        'type': 'clip_by_global_norm',
        'kwargs': {
            'clip_norm': 25.
        }
    }
}
