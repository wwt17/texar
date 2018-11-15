max_epochs = 1000
steps_per_val = 500
steps_per_test = 1917
tau = 1.
n_samples = 10
sample_max_decoding_length = 50
infer_beam_width = 1
infer_max_decoding_length = 50
weight_pg_grd = 0.1
weight_pg_msp = 1. # 0.95

threshold_steps = 10000
minimum_interval_steps = 10000
phases = [
    # (config_data, config_train, mask_pattern)
    ("train_1", "pg_msp", None),
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

train_pg_grd = {
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
    "name": "PG_GRD"
}

train_pg_msp = {
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
    "name": "PG_MSP"
}
