from utils import get_scope_name_of_train_op

max_epochs = int(1e9)
steps_per_eval = 500

infer_beam_width = 5
infer_max_decoding_length = 30

train = {
    'MLE': {
        'optimizer': {
            'type': 'AdamOptimizer',
            'kwargs': {
                'learning_rate': 1e-3
            }
        },
        'gradient_clip': {
            'type': 'clip_by_global_norm',
            'kwargs': {
                'clip_norm': 5.
            }
        },
    },
}

for name, hparams in train.items():
    hparams['name'] = get_scope_name_of_train_op(name)
