source_vocab_file = 'data/wmt_en-de/vocab.en'
target_vocab_file = 'data/wmt_en-de/vocab.de'

train_0 = {
    'batch_size': 160,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": 'data/wmt_en-de/train.en',
        'vocab_file': source_vocab_file,
        'max_seq_length': 50
    },
    'target_dataset': {
        'files': 'data/wmt_en-de/train.de',
        'vocab_file': target_vocab_file,
        'max_seq_length': 50
    },
}

train_1 = train_0

val = {
    'batch_size': 32,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/wmt_en-de/dev.en',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/wmt_en-de/dev.de',
        'vocab_file': target_vocab_file,
    },
}

test = {
    'batch_size': 32,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/wmt_en-de/test.en',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/wmt_en-de/test.de',
        'vocab_file': target_vocab_file,
    },
}
