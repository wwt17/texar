max_epochs = 1000
steps_per_eval = 500
tau = 1.
infer_beam_width = 1
infer_max_decoding_length = 50
weight_rl = 0.1

threshold_steps = 10000
minimum_interval_steps = 10000
phases = [
    # (config_data, config_train, mask_pattern)
    ("train_0", "rl_xe", None),
]

train_xe_0 = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 1e-3
        }
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {
            "clip_norm": 5.
        }
    },
    "name": "XE_0"
}

train_xe_1 = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 1e-5
        }
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {
            "clip_norm": 5.
        }
    },
    "name": "XE_1"
}

train_debleu_0 = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 1e-5
        }
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {
            "clip_norm": 5.
        }
    },
    "name": "DEBLEU_0"
}

train_debleu_1 = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 1e-6
        }
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {
            "clip_norm": 5.
        }
    },
    "name": "DEBLEU_1"
}

train_rl_xe = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 1e-5
        }
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {
            "clip_norm": 5.
        }
    },
    "name": "RL_XE"
}
