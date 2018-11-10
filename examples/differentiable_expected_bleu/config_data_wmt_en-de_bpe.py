source_vocab_file = 'data/wmt_en-de_bpe/vocab.en'
target_vocab_file = 'data/wmt_en-de_bpe/vocab.de'

train_0 = {
    'batch_size': 128,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": 'data/wmt_en-de_bpe/train.en',
        'vocab_file': source_vocab_file,
        'max_seq_length': 50
    },
    'target_dataset': {
        'files': 'data/wmt_en-de_bpe/train.de',
        'vocab_file': target_vocab_file,
        'max_seq_length': 50
    },
}

train_1 = train_0

val = {
    'batch_size': 32,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/wmt_en-de_bpe/dev.en',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/wmt_en-de_bpe/dev.de',
        'vocab_file': target_vocab_file,
    },
}

test = {
    'batch_size': 32,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/wmt_en-de_bpe/test.en',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/wmt_en-de_bpe/test.de',
        'vocab_file': target_vocab_file,
    },
}
