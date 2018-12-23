"""
Information Retrieval.
Equivalent to the following commands:

cd data2text
GOLD_FILE="../nba_data/gold.valid.txt"
GEN_FILE="../attn_copynet_sd_path/ckpt/hypos.step96860.val.txt"
INTER_FILE="${GEN_FILE}.h5"
GPUID=1
python data_utils.py -mode prep_gen_data -dict_pfx roto-ie -input_path rotowire -val_file ${GOLD_FILE} -gen_fi ${GEN_FILE} -output_fi ${INTER_FILE}
th extractor.lua -gpuid ${GPUID} -datafile roto-ie.h5 -dict_pfx roto-ie -just_eval -preddata ${INTER_FILE}
python non_rg_metrics.py ${GOLD_FILE} ${INTER_FILE}-tuples.txt
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import subprocess
from data2text.data_utils import prep_generated_data
from data2text.non_rg_metrics import calc_precrec, get_items

data2text_dir = "data2text"


def get_precrec(
        gold_file, gen_file, inter_file, gpuid=0,
        dict_pfx=os.path.join(data2text_dir, "roto-ie"),
        train_file=os.path.join(data2text_dir, "rotowire", "train.json")):
    prep_generated_data(gen_file, dict_pfx, inter_file,
                        train_file=train_file, val_file=gold_file)

    ret = subprocess.call(
        ["th", "extractor.lua",
         "-gpuid", str(gpuid+1),
         "-datafile", os.path.abspath("{}.h5".format(dict_pfx)),
         "-dict_pfx", os.path.abspath(dict_pfx),
         "-just_eval",
         "-preddata", os.path.abspath(inter_file)],
        cwd=data2text_dir)
    if ret != 0:
        raise Exception(
            "run extractor.lua failed with return value {}".format(ret))

    pred_file = "{}-tuples.txt".format(inter_file)
    gold_items, pred_items = map(get_items, (gold_file, pred_file))
    return calc_precrec(gold_items, pred_items)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--gold_file")
    argparser.add_argument("--gen_file")
    argparser.add_argument("--inter_file", default="")
    argparser.add_argument("--gpuid", type=int, default=0)
    args = argparser.parse_args()
    if not args.inter_file:
        s = args.gen_file
        suffix = ".txt"
        if s.endswith(suffix):
            s = s[:-len(suffix)]
        args.inter_file = s + ".h5"
    prec, rec = get_precrec(**vars(args))
