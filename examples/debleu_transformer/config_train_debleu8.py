max_epochs = 1000
steps_per_val = 2000
steps_per_test = 10000

threshold_steps = 20000
minimum_interval_steps = 10000
phases = [
    # (config_data, config_train, mask_pattern)
    ("train", "xe", None),
    ("train", "debleu", (2, 2)),
    ("train", "debleu", (4, 2)),
    ("train", "debleu", (1, 0)),
]

max_order = 8
weights=[1. / max_order for _ in range(max_order)]

loss_label_confidence = 0.9
tau = 1.

infer_max_decoding_length = 200
infer_beam_width = 1
infer_alpha = 0.6

n_samples = 1
sample_max_decoding_length = 200
greedy_max_decoding_length = 200

weight_pg_grd = 1.
weight_pg_msp = 1.

train_xe = {
    'name': 'xe',
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'beta1': 0.9,
            'beta2': 0.997,
            'epsilon': 1e-9
        }
    },
}

train_debleu = {
    'name': 'debleu',
    'optimizer': {
        'type': 'GradientDescentOptimizer',
        'kwargs': {
            'learning_rate': 1e-4
        }
    },
    'gradient_clip': {
        'type': 'clip_by_global_norm',
        'kwargs': {
            'clip_norm': 25.
        }
    },
}

train_pg_grd = {
    'name': 'pg_grd',
    'optimizer': {
        'type': 'GradientDescentOptimizer',
        'kwargs': {
            'learning_rate': 1e-4
        }
    },
    'gradient_clip': {
        'type': 'clip_by_global_norm',
        'kwargs': {
            'clip_norm': 25.
        }
    },
}

train_pg_msp = {
    'name': 'pg_msp',
    'optimizer': {
        'type': 'GradientDescentOptimizer',
        'kwargs': {
            'learning_rate': 1e-4
        }
    },
    'gradient_clip': {
        'type': 'clip_by_global_norm',
        'kwargs': {
            'clip_norm': 25.
        }
    },
}

d_model = 512

lr = {
    'learning_rate_schedule': 'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
    'lr_constant': 2 * (d_model ** -0.5),
    'static_lr': 1e-3,
    'warmup_steps': 16000,
}
