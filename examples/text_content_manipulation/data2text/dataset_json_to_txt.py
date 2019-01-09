import json
import argparse

stages = ['train', 'valid', 'test']
sent_fields = {
    'sent': 'target',
}
sd_fields = {
    'entry': 'value',
    'attribute': 'rel',
    'value': 'entity',
}
ref_strs = {
    '': 'query',
    '_ref': 'retrieved',
}


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--json_prefix', default='retrieved_')
    argparser.add_argument('--txt_prefix', default='nba_data/nba.')
    argparser.add_argument('--txt_suffix', default='.txt')
    args = argparser.parse_args()
    for stage in stages:
        with open('{}{}.json'.format(args.json_prefix, stage)) as json_file:
            dataset = json.load(json_file)
        def txt_file_name_of(field, ref_str):
            return '{}{}{}.{}{}'.format(args.txt_prefix, field, ref_str, stage, args.txt_suffix)
        all_files = {ref_str: [{field: open(txt_file_name_of(field, ref_str), 'w') for field in fields} for fields in (sent_fields, sd_fields)] for ref_str in ref_strs}
        for data in dataset:
            for ref_str, ref_key in ref_strs.items():
                sent_files, sd_files = all_files[ref_str]
                ref_data = data[ref_key]
                for field, key in sent_fields.items():
                    print(ref_data[key], file=sent_files[field])
                records = ref_data['records']
                for field, key in sd_fields.items():
                    print(' '.join(str(record[key]) for record in records), file=sd_files[field])
        for sent_sd_files in all_files.values():
            for files in sent_sd_files:
                for file in files.values():
                    file.close()
