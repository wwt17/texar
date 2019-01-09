from __future__ import print_function
from __future__ import unicode_literals

from texar.data import make_vocab as make_vocab

try:
    import codecs
    def open(*args, **kwargs):
        return codecs.open(encoding='utf-8', *args, **kwargs)
except ImportError:
    pass

prefix = 'nba.'
suffix = '.vocab.txt'
ref_strs = ['', '_ref']
stages = ['train']
fields = ['sent', 'entry', 'attribute', 'value']

if __name__ == '__main__':
    vocabs = {}
    for field in fields:
        filenames = []
        for stage in stages:
            for ref_str in ref_strs:
                filenames.append('{prefix}{field}{ref_str}.{stage}.txt'.format(prefix=prefix, field=field, ref_str=ref_str, stage=stage))
        print('files: {}'.format(' '.join(filenames)))
        vocabs[field] = list(make_vocab(filenames))

    vocabs['attribute'].sort()
    vocabs['value'].sort()
    f = lambda s: str(s).isdigit()
    numbers, others = list(filter(f, vocabs['entry'])), list(filter(lambda s: not f(s), vocabs['entry']))
    f = lambda s: s in vocabs['value']
    entities, others = list(filter(f, others)), list(filter(lambda s: not f(s), others))
    numbers.sort(key=int)
    entities.sort(key=vocabs['value'].index)
    others.sort()
    print('others:', others)
    vocabs['entry'] = numbers + entities + others
    vocabs['sent'].sort()

    all_vocab = set()
    for field in fields:
        with open('{prefix}{field}{suffix}'.format(prefix=prefix, field=field, suffix=suffix), 'w') as vocab_file:
            vocab_file.write(u'\n'.join(vocabs[field]))
            all_vocab.update(vocabs[field])

    def get_key(token):
        if token in vocabs['entry']:
            return (0, vocabs['entry'].index(token))
        if token in vocabs['attribute']:
            return (1, vocabs['attribute'].index(token))
        return (2, token)
    all_vocab = sorted(all_vocab, key=get_key)
    with open('{prefix}{field}{suffix}'.format(prefix=prefix, field='all', suffix=suffix), 'w') as vocab_file:
        vocab_file.write(u'\n'.join(all_vocab))
