"""Configurations of Transformer model
"""
import copy
import texar as tx

random_seed = 1
beam_width = 1
alpha = 0.6
n_layers = 6
n_heads = 8
d_k = 64
d_v = 64
d_model = 512
d_inner = 512 * 4
dropout = 0.1

assert d_k == d_v

emb = {
    'name': 'lookup_table',
    'dim': d_model,
    'initializer': {
        'type': 'random_normal_initializer',
        'kwargs': {
            'mean': 0.0,
            'stddev': hidden_dim**-0.5,
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

loss_label_confidence = 0.9

opt = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'beta1': 0.9,
            'beta2': 0.997,
            'epsilon': 1e-9
        }
    }
}

lr = {
    'learning_rate_schedule': 'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
    'lr_constant': 2 * (hidden_dim ** -0.5),
    'static_lr': 1e-3,
    'warmup_steps': 16000,
}
