from __future__ import print_function
import sys, codecs, json, os
from collections import Counter, defaultdict, namedtuple
import copy
from nltk import sent_tokenize, word_tokenize
import numpy as np
import h5py
# import re
import random
import math
from text2num import text2num, NumberException
import argparse

def open(*args, **kwargs):
    return codecs.open(encoding="utf-8", *args, **kwargs)

random.seed(2)

Ent = namedtuple("Ent", ["start", "end", "s", "is_pron"])
Num = namedtuple("Num", ["start", "end", "s"])
Rel = namedtuple("Rel", ["ent", "num", "type", "aux"])
stuff_names = ('sent', 'len', 'entdist', 'numdist', 'label')

prons = {"he", "He", "him", "Him", "his", "His", "they", "They", "them", "Them", "their", "Their"}  # leave out "it"
singular_prons = {"he", "He", "him", "Him", "his", "His"}
plural_prons = {"they", "They", "them", "Them", "their", "Their"}

number_words = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
                "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"}


def get_ents(dat):
    players = set()
    teams = set()
    cities = set()
    team_strs = ["vis", "home"]
    for thing in dat:
        for team_str in team_strs:
            names = thing["{}_name".format(team_str)], thing["{}_line".format(team_str)]["TEAM-NAME"]
            prefixes = ["", thing["{}_city".format(team_str)] + " "]
            for prefix in prefixes:
                for name in names:
                    teams.add(prefix + name)
        # special case for this
        for team_str in team_strs:
            if thing["{}_city".format(team_str)] == "Los Angeles":
                teams.add("LA" + thing["{}_name".format(team_str)])
        # sometimes team_city is different
        for team_str in team_strs:
            cities.add(thing["{}_city".format(team_str)])
        players.update(thing["box_score"]["PLAYER_NAME"].values())
        cities.update(thing["box_score"]["TEAM_CITY"].values())

    for entset in [players, teams, cities]:
        for k in list(entset):
            pieces = k.split()
            if len(pieces) > 1:
                for piece in pieces:
                    if len(piece) > 1 and piece not in ["II", "III", "Jr.", "Jr"]:
                        entset.add(piece)

    all_ents = players | teams | cities

    return all_ents, players, teams, cities


def deterministic_resolve(pron, players, teams, cities, curr_ents, prev_ents, max_back=1):
    # we'll just take closest compatible one.
    # first look in current sentence; if there's an antecedent here return None, since
    # we'll catch it anyway
    for j in xrange(len(curr_ents) - 1, -1, -1):
        if pron in singular_prons and curr_ents[j][2] in players:
            return None
        elif pron in plural_prons and curr_ents[j][2] in teams:
            return None
        elif pron in plural_prons and curr_ents[j][2] in cities:
            return None

    # then look in previous max_back sentences
    if len(prev_ents) > 0:
        for i in xrange(len(prev_ents) - 1, len(prev_ents) - 1 - max_back, -1):
            for j in xrange(len(prev_ents[i]) - 1, -1, -1):
                if pron in singular_prons and prev_ents[i][j][2] in players:
                    return prev_ents[i][j]
                elif pron in plural_prons and prev_ents[i][j][2] in teams:
                    return prev_ents[i][j]
                elif pron in plural_prons and prev_ents[i][j][2] in cities:
                    return prev_ents[i][j]
    return None


def extract_entities(sent, all_ents, prons, prev_ents=None, resolve_prons=False,
                     players=None, teams=None, cities=None):
    sent_ents = []
    i = 0
    while i < len(sent):
        if sent[i] in prons:
            if resolve_prons:
                referent = deterministic_resolve(sent[i], players, teams, cities, sent_ents, prev_ents)
                if referent is None:
                    sent_ents.append(Ent(i, i + 1, sent[i], True))  # is a pronoun
                else:
                    # print("replacing", sent[i], "with", referent[2], "in", " ".join(sent))
                    sent_ents.append(
                        Ent(i, i + 1, referent[2], False))  # pretend it's not a pron and put in matching string
            else:
                sent_ents.append(Ent(i, i + 1, sent[i], True))  # is a pronoun
            i += 1
        elif sent[i] in all_ents:  # findest longest spans; only works if we put in words...
            j = 1
            while i + j <= len(sent) and " ".join(sent[i:i + j]) in all_ents:
                j += 1
            sent_ents.append(Ent(i, i + j - 1, " ".join(sent[i:i + j - 1]), False))
            i += j - 1
        else:
            i += 1
    return sent_ents


def annoying_number_word(sent, i):
    ignores = {"three point", "three - point", "three - pt", "three pt", "three - pointer",
               "three - pointers", "three pointers", "three - points"}
    return " ".join(sent[i:i + 3]) not in ignores and " ".join(sent[i:i + 2]) not in ignores


def extract_numbers(sent):
    sent_nums = []
    i = 0
    while i < len(sent):
        toke = sent[i]
        a_number = False
        try:
            itoke = int(toke)
            a_number = True
        except ValueError:
            pass
        if a_number:
            sent_nums.append(Num(i, i + 1, int(toke)))
            i += 1
        elif toke in number_words and annoying_number_word(sent, i):  # get longest span  (this is kind of stupid)
            j = 1
            while i + j < len(sent) and sent[i + j] in number_words and annoying_number_word(sent, i + j):
                j += 1
            try:
                sent_nums.append(Num(i, i + j, text2num(" ".join(sent[i:i + j]))))
            except NumberException:
                sent_nums.append(Num(i, i + 1, text2num(sent[i])))
            i += j
        else:
            i += 1
    return sent_nums


def get_player_idx(bs, entname):
    keys = []
    for k, v in bs["PLAYER_NAME"].iteritems():
        if entname == v:
            keys.append(k)
    if len(keys) == 0:
        for k, v in bs["SECOND_NAME"].iteritems():
            if entname == v:
                keys.append(k)
        if len(keys) > 1:  # take the earliest one
            keys.sort(key=lambda x: int(x))
            keys = keys[:1]
            # print("picking", bs["PLAYER_NAME"][keys[0]])
    if len(keys) == 0:
        for k, v in bs["FIRST_NAME"].iteritems():
            if entname == v:
                keys.append(k)
        if len(keys) > 1:  # if we matched on first name and there are a bunch just forget about it
            return None
    # if len(keys) == 0:
    # print("Couldn't find", entname, "in", bs["PLAYER_NAME"].values())
    assert len(keys) <= 1, entname + " : " + str(bs["PLAYER_NAME"].values())
    return keys[0] if len(keys) > 0 else None


def get_rels(entry, ents, nums, players, teams, cities):
    """
    this looks at the box/line score and figures out which (entity, number) pairs
    are candidate true relations, and which can't be.
    if an ent and number don't line up (i.e., aren't in the box/line score together),
    we give a NONE label, so for generated summaries that we extract from, if we predict
    a label we'll get it wrong (which is presumably what we want).
    N.B. this function only looks at the entity string (not position in sentence), so the
    string a pronoun corefers with can be snuck in....
    """
    rels = []
    bs = entry["box_score"]
    for i, ent in enumerate(ents):
        if ent.is_pron:  # pronoun
            continue  # for now
        entname = ent.s
        # assume if a player has a city or team name as his name, they won't use that one (e.g., Orlando Johnson)
        if entname in players and entname not in cities and entname not in teams:
            pidx = get_player_idx(bs, entname)
            for j, numtup in enumerate(nums):
                found = False
                strnum = str(numtup.s)
                if pidx is not None:  # player might not actually be in the game or whatever
                    for colname, col in bs.iteritems():
                        if col[pidx] == strnum:  # allow multiple for now
                            rels.append(Rel(ent, numtup, "PLAYER-" + colname, pidx))
                            found = True
                if not found:
                    rels.append(Rel(ent, numtup, "NONE", None))

        else:  # has to be city or team
            entpieces = entname.split()
            linescore = None
            is_home = None
            if entpieces[0] in entry["home_city"] or entpieces[-1] in entry["home_name"]:
                linescore = entry["home_line"]
                is_home = True
            elif entpieces[0] in entry["vis_city"] or entpieces[-1] in entry["vis_name"]:
                linescore = entry["vis_line"]
                is_home = False
            elif "LA" in entpieces[0]:
                if entry["home_city"] == "Los Angeles":
                    linescore = entry["home_line"]
                    is_home = True
                elif entry["vis_city"] == "Los Angeles":
                    linescore = entry["vis_line"]
                    is_home = False
            for j, numtup in enumerate(nums):
                found = False
                strnum = str(numtup.s)
                if linescore is not None:
                    for colname, val in linescore.iteritems():
                        if val == strnum:
                            # rels.append(Rel(ent, numtup, "TEAM-" + colname, is_home))
                            # apparently I appended TEAM- at some pt...
                            rels.append(Rel(ent, numtup, colname, is_home))
                            found = True
                if not found:
                    rels.append(Rel(ent, numtup, "NONE", None))  # should i specialize the NONE labels too?
    return rels


def get_candidate_rels(entry, summ, all_ents, prons, players, teams, cities):
    """
    generate tuples of form (sentence_tokens, [rels]) to candrels
    """
    sents = sent_tokenize(summ)
    for j, sent in enumerate(sents):
        # tokes = word_tokenize(sent)
        tokes = sent.split()
        ents = extract_entities(tokes, all_ents, prons)
        nums = extract_numbers(tokes)
        rels = get_rels(entry, ents, nums, players, teams, cities)
        if len(rels) > 0:
            yield (tokes, rels)


stages = ["train", "valid", "test"]


def get_datasets(path="rotowire"):
    datasets = {}
    for stage in stages:
        with open(os.path.join(path, "{}.json".format(stage)), "r") as f:
            datasets[stage] = json.load(f)

    all_ents, players, teams, cities = get_ents(datasets["train"])

    extracted_stuff = {}
    for stage, dataset in datasets.items():
        nugz = []
        for i, entry in enumerate(dataset):
            summ = " ".join(entry['summary'])
            nugz.extend(get_candidate_rels(entry, summ, all_ents, prons, players, teams, cities))
        extracted_stuff[stage] = nugz

    return extracted_stuff


def get_to_data(tup, vocab, labeldict, max_len):
    """
    tup is (sent, [rels]);
    each rel is ((ent_start, ent_ent, ent_str), (num_start, num_end, num_str), label)
    """
    sent = [vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in tup[0]]
    sentlen = len(sent)
    sent.extend([-1] * (max_len - sentlen))
    for rel in tup[1]:
        ent, num, label, idthing = rel
        ent_dists = [j - ent[0] if j < ent[0] else j - ent[1] + 1 if j >= ent[1] else 0 for j in xrange(max_len)]
        num_dists = [j - num[0] if j < num[0] else j - num[1] + 1 if j >= num[1] else 0 for j in xrange(max_len)]
        yield sent, sentlen, ent_dists, num_dists, labeldict[label]


def get_multilabeled_data(tup, vocab, labeldict, max_len):
    """
    used for val, since we have contradictory labelings...
    tup is (sent, [rels]);
    each rel is ((ent_start, ent_end, ent_str), (num_start, num_end, num_str), label)
    """
    sent = [vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in tup[0]]
    sentlen = len(sent)
    sent.extend([-1] * (max_len - sentlen))
    # get all the labels for the same rel
    unique_rels = defaultdict(list)
    for rel in tup[1]:
        ent, num, label, idthing = rel
        unique_rels[ent, num].append(label)

    for rel, label_list in unique_rels.iteritems():
        ent, num = rel
        ent_dists = [j - ent[0] if j < ent[0] else j - ent[1] + 1 if j >= ent[1] else 0 for j in xrange(max_len)]
        num_dists = [j - num[0] if j < num[0] else j - num[1] + 1 if j >= num[1] else 0 for j in xrange(max_len)]
        yield sent, sentlen, ent_dists, num_dists, [labeldict[label] for label in label_list]


def append_labelnums(labels):
    max_num_labels = max(map(len, labels))
    print("max num labels", max_num_labels)

    # append number of labels to labels
    for i, labellist in enumerate(labels):
        l = len(labellist)
        labellist.extend([-1] * (max_num_labels - l))
        labellist.append(l)


def preprocess_data(data):
    tokens, rels = data
    tgt_ranges = set()
    new_rels = []
    for rel in rels:
        new_rels.append(Rel(
            Ent(rel.ent.start, rel.ent.end,
                unicode(rel.ent.s).replace(' ', '_'), rel.ent.is_pron),
            rel.num, rel.type, rel.aux))
        for e in [rel.ent, rel.num]:
            tgt_ranges.add((e.start, e.end))

    # process start and end idxs
    for rel_i, rel in enumerate(new_rels):
        new_rel = []
        for i in range(2):
            offset = 0
            for start, end in tgt_ranges:
                if rel[i].start >= end:
                    offset += end - start - 1
            start = rel[i].start - offset
            end = start + 1
            new_e = type(rel[i])(*((start, end) + rel[i][2:]))
            new_rel.append(new_e)
        new_rel = Rel(*(tuple(new_rel) + rel[2:]))
        new_rels[rel_i] = new_rel

    # process target tokens to connect multiword with underscore
    new_tokens = []
    for idx, word in enumerate(tokens):
        between = False
        for start, end in tgt_ranges:
            if idx == start:
                new_tokens.append(u'_'.join(tokens[start:end]))
                between = True
                break
            elif start < idx < end:
                between = True
        if not between:
            new_tokens.append(word)
        else:
            continue
    return new_tokens, new_rels

def preprocess_dataset(dataset):
    return list(filter(
        lambda data: len(data[0]) <= 50 and len(data[1]) <= 50,
        map(preprocess_data, dataset)))


# modified full sentence IE training
def save_full_sent_data(outfile, path="rotowire", multilabel_train=False, nonedenom=0, backup=False, verbose=True):
    datasets = get_datasets(path)
    if not backup:
        datasets = {stage: preprocess_dataset(dataset) for stage, dataset in datasets.items()}
    # make vocab and get labels
    word_counter = Counter()
    [word_counter.update(tup[0]) for tup in datasets['train']]
    for k in word_counter.keys():
        if word_counter[k] < 2:
            del word_counter[k]  # will replace w/ unk
    word_counter["UNK"] = 1
    vocab = {wrd: i + 1 for i, wrd in enumerate(word_counter.keys())}
    labelset = set()
    [labelset.update(rel.type for rel in tup[1]) for tup in datasets['train']]
    labeldict = {label: i + 1 for i, label in enumerate(labelset)}

    # save stuff
    stuffs = {stage: [] for stage in datasets}

    max_trlen = max(len(tup[0]) for tup in datasets['train'])
    print("max tr sentence length:", max_trlen)

    # do training data
    for tup in datasets['train']:
        stuffs['train'].extend((get_multilabeled_data if multilabel_train else get_to_data)(tup, vocab, labeldict, max_trlen))

    if multilabel_train:
        append_labelnums([x[-1] for x in stuffs['train']])

    if nonedenom > 0:
        # don't keep all the NONE labeled things
        trlabels = [x[-1] for x in stuffs['train']]
        none_idxs = [i for i, labellist in enumerate(trlabels) if labellist[0] == labeldict["NONE"]]
        random.shuffle(none_idxs)
        # allow at most 1/(nonedenom+1) of NONE-labeled
        num_to_keep = int(math.floor(float(len(trlabels) - len(none_idxs)) / nonedenom))
        print("originally", len(trlabels), "training examples")
        print("keeping", num_to_keep, "NONE-labeled examples")
        ignore_idxs = set(none_idxs[num_to_keep:])

        # get rid of most of the NONE-labeled examples
        stuffs['train'] = [thing for i, thing in enumerate(stuffs['train']) if i not in ignore_idxs]

    print(len(stuffs['train']), "training examples")

    if verbose:
        for _ in stuffs['train'][0]:
            print(_)

    for stage in ['valid', 'test']:
        # do val/test, which we also consider multilabel
        dataset = datasets[stage]
        max_len = max(len(tup[0]) for tup in dataset)
        for tup in dataset:
            stuffs[stage].extend(
                get_multilabeled_data(tup, vocab, labeldict, max_len))

        append_labelnums([x[-1] for x in stuffs[stage]])

        print(len(stuffs[stage]), "{} examples".format(stage))

    stage_to_abbr = {"train": "tr", "valid": "val", "test": "test"}
    h5fi = h5py.File(outfile, "w")
    for stage, stuff in stuffs.items():
        abbr = stage_to_abbr[stage]
        for name, content in zip(stuff_names, zip(*stuff)):
            h5fi["{}{}s".format(abbr, name)] = np.array(content, dtype=int)
    h5fi.close()

    # write dicts
    for d, name in ((vocab, 'dict'), (labeldict, 'labels')):
        revd = {v: k for k, v in d.iteritems()}
        with open("{}.{}".format(outfile.split('.')[0], name), "w+") as f:
            for i in xrange(1, len(revd) + 1):
                f.write("%s %d \n" % (revd[i], i))


def convert_aux(additional):
    aux = 'NONE'
    if additional is None:
        aux = "NONE"
    elif isinstance(additional, unicode):
        aux = str(additional)
    elif isinstance(additional, bool):
        if additional:
            aux = "HOME"
        else:
            aux = "AWAY"
    return aux


def write_data_to_line(data, outfile, filter_none=True):
    sent = ' '.join(data[0])
    rels = data[1]
    for rel in rels:
        if filter_none and rel[2] == "NONE":
            continue
        additional = rel[3]
        aux = convert_aux(additional)
        line_to_write = u'\t'.join(map(unicode, (sent, rel[0][0], rel[0][1], rel[0][2],
                                                 rel[1][0], rel[1][1], rel[1][2], rel[2], aux)))
        outfile.write(line_to_write + '\n')


def split_sent_to_triples(dataset, filter_none=True):
    processed_data = []
    for data in dataset:
        rels = data[1]
        triples = set()
        for rel in rels:
            if filter_none and rel[2] == "NONE":
                continue
            additional = rel[3]
            aux = convert_aux(additional)
            triples.add((rel[2], rel[0][2], str(rel[1][2])))
            # add name information for copying
            if "PLAYER" in rel[2]:
                triples.add(("PLAYER_NAME", rel[0][2], rel[0][2]))
            elif "TEAM" in rel[2]:
                triples.add(("TEAM_NAME", rel[0][2], rel[0][2]))
            # add home away information
            # if aux in ("HOME", "AWAY"):
            #     triples.add(("HOME_AWAY", rel[0][2], aux))
        if len(triples) > 0:
            processed_data.append({'tokens': data[0], 'triples': list(triples)})
    return processed_data


def make_translate_corpus(dataset):
    src_lines = []
    tgt_lines = []
    for data in dataset:
        rels = data[1]
        tokens = data[0]
        triples = set()
        tgt_ranges = set()
        for rel in rels:
            if rel[2] == "NONE":
                continue
            rel_type = rel[2]
            ent_start = int(rel[0][0])
            ent_end = int(rel[0][1])
            ent = unicode(rel[0][2]).replace(' ', '_')
            val_start = int(rel[1][0])
            val_end = int(rel[1][1])
            val = unicode(rel[1][2]).replace(' ', '_')
            if ent_end - ent_start > 1:
                tgt_ranges.add((ent_start, ent_end))
            if val_end - val_start > 1:
                tgt_ranges.add((val_start, val_end))

            additional = rel[3]
            aux = convert_aux(additional)
            triples.add((rel_type, ent, val))
            # add name information for copying
            if "PLAYER" in rel_type:
                triples.add(("PLAYER_NAME", ent, ent))
            elif "TEAM" in rel_type:
                triples.add(("TEAM_NAME", ent, ent))
            # add home away information
            # if aux in ("HOME", "AWAY"):
            #     triples.add(("HOME_AWAY", ent, aux))

        if len(triples) == 0:
            continue
        # process target tokens to connect multiword with underscore
        new_tokens = []
        for idx, word in enumerate(tokens):
            between = False
            for start, end in tgt_ranges:
                if idx == start:
                    new_tokens.append(u'_'.join(tokens[start:end]))
                    between = True
                    break
                elif start < idx < end:
                    between = True
            if not between:
                new_tokens.append(word)
            else:
                continue
        sorted_triples = list(sorted(triples, key=lambda x: x[0]))
        if len(sorted_triples) > 50 or len(new_tokens) > 50:
            continue
        src_line = u''
        for t in sorted_triples:
            src_line += u'|'.join((t[2], t[0], t[1])) + u' '
        tgt_line = u' '.join(new_tokens)
        src_lines.append(src_line)
        tgt_lines.append(tgt_line)

    return src_lines, tgt_lines


# for extracting sentence-data pairs
def extract_sentence_data(outfile, path="rotowire"):
    datasets = get_datasets(path)
    for stage, dataset in datasets.items():
        # output json
        with open(outfile + '.{}.json'.format(stage), 'w') as of:
            json.dump(split_sent_to_triples(dataset), of)

        # output translate data files
        with open(outfile + '.{}.src'.format(stage), 'w') as of1, \
                open(outfile + '.{}.tgt'.format(stage), 'w') as of2:
            src_lines, tgt_lines = make_translate_corpus(dataset)
            of1.write(u'\n'.join(src_lines))
            of2.write(u'\n'.join(tgt_lines))

        # output csv
        with open(outfile + '.{}'.format(stage), 'w') as of:
            for data in dataset:
                write_data_to_line(data, of)


def prep_generated_data(genfile, dict_pfx, outfile, train_file, val_file, backup=False):
    # recreate vocab and labeldict
    def read_dict(s):
        d = {}
        with open("{}.{}".format(dict_pfx, s), "r") as f:
            for line in f:
                pieces = line.strip().split()
                d[pieces[0]] = int(pieces[1])
        return d

    vocab, labeldict = map(read_dict, ["dict", "labels"])

    with open(genfile, "r") as f:
        gens = f.readlines()

    with open(train_file, "r") as f:
        trdata = json.load(f)

    all_ents, players, teams, cities = get_ents(trdata)

    if not backup:
        all_ents = set(x.replace(' ', '_') for x in all_ents)

    with open(val_file, "r") as f:
        valdata = json.load(f) if backup else [[x.split('|') for x in line.split()] for line in f]

    assert len(valdata) == len(gens), "len(valdata) = {}, len(gens) = {}".format(len(valdata), len(gens))

    # extract ent-num pairs from generated sentence
    nugz = []  # to hold (sentence_tokens, [rels]) tuples
    if not backup:
        for entry, summ in zip(valdata, gens):
            gold_rels = []
            for rel in entry:
                if rel[1] not in ("TEAM_NAME", "PLAYER_NAME"):
                    gold_rels.append((rel[2], int(rel[0]), rel[1]))

            sent = summ.split()
            ents = extract_entities(sent, all_ents, prons)
            nums = extract_numbers(sent)
            extracted_rels = []
            for ent in ents:
                for num in nums:
                    match = False
                    for rel in gold_rels:
                        if ent[2] == rel[0] and num[2] == rel[1]:
                            match = True
                            extracted_rels.append(Rel(ent, num, rel[2], None))
                    if not match:
                        extracted_rels.append(Rel(ent, num, 'NONE', None))
            nugz.append((sent, extracted_rels))
    else:
        sent_reset_indices = {0}  # sentence indices where a box/story is reset
        for entry, summ in zip(valdata, gens):
            nugz.extend(get_candidate_rels(entry, summ, all_ents, prons, players, teams, cities))
            sent_reset_indices.add(len(nugz))

    # save stuff
    max_len = max(len(tup[0]) for tup in nugz)
    p = []

    rel_reset_indices = []
    for t, tup in enumerate(nugz):
        if not backup or t in sent_reset_indices:  # then last rel is the last of its box
            rel_reset_indices.append(len(p))
        p.extend(get_multilabeled_data(tup, vocab, labeldict, max_len))

    append_labelnums([x[-1] for x in p])

    print(len(p), "prediction examples")

    h5fi = h5py.File(outfile, "w")
    for name, content in zip(stuff_names, zip(*p)):
        h5fi["val{}s".format(name)] = np.array(content, dtype=int)
    h5fi["boxrestartidxs"] = np.array(np.array(rel_reset_indices), dtype=int)  # 1-indexed
    h5fi.close()


################################################################################

bs_keys = ["PLAYER-PLAYER_NAME", "PLAYER-START_POSITION", "PLAYER-MIN", "PLAYER-PTS",
           "PLAYER-FGM", "PLAYER-FGA", "PLAYER-FG_PCT", "PLAYER-FG3M", "PLAYER-FG3A",
           "PLAYER-FG3_PCT", "PLAYER-FTM", "PLAYER-FTA", "PLAYER-FT_PCT", "PLAYER-OREB",
           "PLAYER-DREB", "PLAYER-REB", "PLAYER-AST", "PLAYER-TO", "PLAYER-STL", "PLAYER-BLK",
           "PLAYER-PF", "PLAYER-FIRST_NAME", "PLAYER-SECOND_NAME"]

ls_keys = ["TEAM-PTS_QTR1", "TEAM-PTS_QTR2", "TEAM-PTS_QTR3", "TEAM-PTS_QTR4",
           "TEAM-PTS", "TEAM-FG_PCT", "TEAM-FG3_PCT", "TEAM-FT_PCT", "TEAM-REB",
           "TEAM-AST", "TEAM-TOV", "TEAM-WINS", "TEAM-LOSSES", "TEAM-CITY", "TEAM-NAME"]

NUM_PLAYERS = 13


def get_player_idxs(entry):
    nplayers = 0
    home_players, vis_players = [], []
    for k, v in entry["box_score"]["PTS"].iteritems():
        nplayers += 1

    num_home, num_vis = 0, 0
    for i in xrange(nplayers):
        player_city = entry["box_score"]["TEAM_CITY"][str(i)]
        if player_city == entry["home_city"]:
            if len(home_players) < NUM_PLAYERS:
                home_players.append(str(i))
                num_home += 1
        else:
            if len(vis_players) < NUM_PLAYERS:
                vis_players.append(str(i))
                num_vis += 1
    return home_players, vis_players


def box_preproc2(trdata):
    """
    just gets src for now
    """
    srcs = [[] for i in xrange(2 * NUM_PLAYERS + 2)]

    for entry in trdata:
        home_players, vis_players = get_player_idxs(entry)
        for ii, player_list in enumerate([home_players, vis_players]):
            for j in xrange(NUM_PLAYERS):
                src_j = []
                player_key = player_list[j] if j < len(player_list) else None
                for k, key in enumerate(bs_keys):
                    rulkey = key.split('-')[1]
                    val = entry["box_score"][rulkey][player_key] if player_key is not None else "N/A"
                    src_j.append(val)
                srcs[ii * NUM_PLAYERS + j].append(src_j)

        home_src, vis_src = [], []
        for k in xrange(len(bs_keys) - len(ls_keys)):
            home_src.append("PAD")
            vis_src.append("PAD")

        for k, key in enumerate(ls_keys):
            home_src.append(entry["home_line"][key])
            vis_src.append(entry["vis_line"][key])

        srcs[-2].append(home_src)
        srcs[-1].append(vis_src)

    return srcs


def linearized_preproc(srcs):
    """
    maps from a num-rows length list of lists of ntrain to an
    ntrain-length list of concatenated rows
    """
    lsrcs = []
    for i in xrange(len(srcs[0])):
        src_i = []
        for j in xrange(len(srcs)):
            src_i.extend(srcs[j][i][1:])  # b/c in lua we ignore first thing
        lsrcs.append(src_i)
    return lsrcs


def fix_target_idx(summ, assumed_idx, word, neighborhood=5):
    """
    tokenization can mess stuff up, so look around
    """
    for i in xrange(1, neighborhood + 1):
        if assumed_idx + i < len(summ) and summ[assumed_idx + i] == word:
            return assumed_idx + i
        elif 0 <= assumed_idx - i < len(summ) and summ[assumed_idx - i] == word:
            return assumed_idx - i
    return None


# for each target word want to know where it could've been copied from
def make_pointerfi(outfi, inp_file="rotowire/train.json", resolve_prons=False):
    """
    N.B. this function only looks at string equality in determining pointerness.
    this means that if we sneak in pronoun strings as their referents, we won't point to the
    pronoun if the referent appears in the table; we may use this tho to point to the correct number
    """
    with open(inp_file, "r") as f:
        trdata = json.load(f)

    rulsrcs = linearized_preproc(box_preproc2(trdata))

    all_ents, players, teams, cities = get_ents(trdata)

    skipped = 0

    train_links = []
    for i, entry in enumerate(trdata):
        home_players, vis_players = get_player_idxs(entry)
        inv_home_players = {pkey: jj for jj, pkey in enumerate(home_players)}
        inv_vis_players = {pkey: (jj + NUM_PLAYERS) for jj, pkey in enumerate(vis_players)}
        summ = " ".join(entry['summary'])
        sents = sent_tokenize(summ)
        words_so_far = 0
        links = []
        prev_ents = []
        for j, sent in enumerate(sents):
            tokes = word_tokenize(sent)  # just assuming this gives me back original tokenization
            ents = extract_entities(tokes, all_ents, prons, prev_ents, resolve_prons,
                                    players, teams, cities)
            if resolve_prons:
                prev_ents.append(ents)
            nums = extract_numbers(tokes)
            # should return a list of (enttup, numtup, rel-name, identifier) for each rel licensed by the table
            rels = get_rels(entry, ents, nums, players, teams, cities)
            for (enttup, numtup, label, idthing) in rels:
                if label != 'NONE':
                    # try to find corresponding words (for both ents and nums)
                    ent_start, ent_end, entspan, _ = enttup
                    num_start, num_end, numspan = numtup
                    if isinstance(idthing, bool):  # city or team
                        # get entity indices if any
                        for k, word in enumerate(tokes[ent_start:ent_end]):
                            src_idx = None
                            if word == entry["home_name"]:
                                src_idx = (2 * NUM_PLAYERS + 1) * (len(bs_keys) - 1) - 1  # last thing
                            elif word == entry["home_city"]:
                                src_idx = (2 * NUM_PLAYERS + 1) * (len(bs_keys) - 1) - 2  # second to last thing
                            elif word == entry["vis_name"]:
                                src_idx = (2 * NUM_PLAYERS + 2) * (len(bs_keys) - 1) - 1  # last thing
                            elif word == entry["vis_city"]:
                                src_idx = (2 * NUM_PLAYERS + 2) * (len(bs_keys) - 1) - 2  # second to last thing
                            if src_idx is not None:
                                targ_idx = words_so_far + ent_start + k
                                if targ_idx >= len(entry["summary"]) or entry["summary"][targ_idx] != word:
                                    targ_idx = fix_target_idx(entry["summary"], targ_idx, word)
                                # print(word, rulsrcs[i][src_idx], entry["summary"][words_so_far + ent_start + k])
                                if targ_idx is None:
                                    skipped += 1
                                else:
                                    assert rulsrcs[i][src_idx] == word and entry["summary"][targ_idx] == word
                                    links.append((src_idx, targ_idx))  # src_idx, target_idx

                        # get num indices if any
                        for k, word in enumerate(tokes[num_start:num_end]):
                            src_idx = None
                            if idthing:  # home, so look in the home row
                                if entry["home_line"][label] == word:
                                    col_idx = ls_keys.index(label)
                                    src_idx = 2 * NUM_PLAYERS * (len(bs_keys) - 1) + len(bs_keys) - len(
                                        ls_keys) + col_idx - 1  # -1 b/c we trim first col
                            else:
                                if entry["vis_line"][label] == word:
                                    col_idx = ls_keys.index(label)
                                    src_idx = (2 * NUM_PLAYERS + 1) * (len(bs_keys) - 1) + len(bs_keys) - len(
                                        ls_keys) + col_idx - 1
                            if src_idx is not None:
                                targ_idx = words_so_far + num_start + k
                                if targ_idx >= len(entry["summary"]) or entry["summary"][targ_idx] != word:
                                    targ_idx = fix_target_idx(entry["summary"], targ_idx, word)
                                # print(word, rulsrcs[i][src_idx], entry["summary"][words_so_far + num_start + k])
                                if targ_idx is None:
                                    skipped += 1
                                else:
                                    assert rulsrcs[i][src_idx] == word and entry["summary"][targ_idx] == word
                                    links.append((src_idx, targ_idx))
                    else:  # players
                        # get row corresponding to this player
                        player_row = None
                        if idthing in inv_home_players:
                            player_row = inv_home_players[idthing]
                        elif idthing in inv_vis_players:
                            player_row = inv_vis_players[idthing]
                        if player_row is not None:
                            # ent links
                            for k, word in enumerate(tokes[ent_start:ent_end]):
                                src_idx = None
                                if word == entry["box_score"]["FIRST_NAME"][idthing]:
                                    src_idx = (player_row + 1) * (len(bs_keys) - 1) - 2  # second to last thing
                                elif word == entry["box_score"]["SECOND_NAME"][idthing]:
                                    src_idx = (player_row + 1) * (len(bs_keys) - 1) - 1  # last thing
                                if src_idx is not None:
                                    targ_idx = words_so_far + ent_start + k
                                    if entry["summary"][targ_idx] != word:
                                        targ_idx = fix_target_idx(entry["summary"], targ_idx, word)
                                    if targ_idx is None:
                                        skipped += 1
                                    else:
                                        assert rulsrcs[i][src_idx] == word and entry["summary"][targ_idx] == word
                                        links.append((src_idx, targ_idx))  # src_idx, target_idx
                            # num links
                            for k, word in enumerate(tokes[num_start:num_end]):
                                src_idx = None
                                if word == entry["box_score"][label.split('-')[1]][idthing]:
                                    src_idx = player_row * (len(bs_keys) - 1) + bs_keys.index(
                                        label) - 1  # subtract 1 because we ignore first col
                                if src_idx is not None:
                                    targ_idx = words_so_far + num_start + k
                                    if targ_idx >= len(entry["summary"]) or entry["summary"][targ_idx] != word:
                                        targ_idx = fix_target_idx(entry["summary"], targ_idx, word)
                                    # print(word, rulsrcs[i][src_idx], entry["summary"][words_so_far + num_start + k])
                                    if targ_idx is None:
                                        skipped += 1
                                    else:
                                        assert rulsrcs[i][src_idx] == word and entry["summary"][targ_idx] == word
                                        links.append((src_idx, targ_idx))

            words_so_far += len(tokes)
        train_links.append(links)
    print("SKIPPED", skipped)

    # collapse multiple links
    trlink_dicts = []
    for links in train_links:
        links_dict = defaultdict(list)
        [links_dict[targ_idx].append(src_idx) for src_idx, targ_idx in links]
        trlink_dicts.append(links_dict)

    # write in fmt:
    # targ_idx,src_idx1[,src_idx...]
    with open(outfi, "w+") as f:
        for links_dict in trlink_dicts:
            targ_idxs = sorted(links_dict.keys())
            fmtd = [",".join([str(targ_idx)] + [str(thing) for thing in set(links_dict[targ_idx])])
                    for targ_idx in targ_idxs]
            f.write("%s\n" % " ".join(fmtd))


# for coref prediction stuff
# we'll use string equality for now
def save_coref_task_data(outfile, inp_file="full_newnba_prepdata2.json"):
    with open(inp_file, "r") as f:
        data = json.load(f)

    all_ents, players, teams, cities = get_ents(data["train"])
    datasets = []

    # labels are nomatch, match, pron
    for dataset in [data["train"], data["valid"]]:
        examples = []
        for i, entry in enumerate(dataset):
            summ = entry["summary"]
            ents = extract_entities(summ, all_ents, prons)
            for j in xrange(1, len(ents)):
                # just get all the words from previous mention till this one starts
                prev_start, prev_end, prev_str, _ = ents[j - 1]
                curr_start, curr_end, curr_str, curr_pron = ents[j]
                # window = summ[prev_start:curr_start]
                window = summ[prev_end:curr_start]
                label = None
                if curr_pron:  # prons
                    label = 3
                else:
                    # label = 2 if prev_str == curr_str else 1
                    label = 2 if prev_str in curr_str or curr_str in prev_str else 1
                examples.append((window, label))
        datasets.append(examples)

    # make vocab and get labels
    word_counter = Counter()
    [word_counter.update(tup[0]) for tup in datasets[0]]
    for k in word_counter.keys():
        if word_counter[k] < 2:
            del word_counter[k]  # will replace w/ unk
    word_counter["UNK"] = 1
    vocab = dict(((wrd, i + 1) for i, wrd in enumerate(word_counter.keys())))
    labeldict = {"NOMATCH": 1, "MATCH": 2, "PRON": 3}

    max_trlen = max((len(tup[0]) for tup in datasets[0]))
    max_vallen = max((len(tup[0]) for tup in datasets[1]))
    print("max sentence lengths:", max_trlen, max_vallen)

    # map words to indices
    trwindows = [[vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in window]
                 + [-1] * (max_trlen - len(window)) for (window, label) in datasets[0]]
    trlabels = [label for (window, label) in datasets[0]]
    valwindows = [[vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in window]
                  + [-1] * (max_vallen - len(window)) for (window, label) in datasets[1]]
    vallabels = [label for (window, label) in datasets[1]]

    print(len(trwindows), "training examples")
    print(len(valwindows), "validation examples")
    print(Counter(trlabels))
    print(Counter(vallabels))

    h5fi = h5py.File(outfile, "w")
    h5fi["trwindows"] = np.array(trwindows, dtype=int)
    h5fi["trlens"] = np.array([len(window) for (window, label) in datasets[0]], dtype=int)
    h5fi["trlabels"] = np.array(trlabels, dtype=int)
    h5fi["valwindows"] = np.array(valwindows, dtype=int)
    h5fi["vallens"] = np.array([len(window) for (window, label) in datasets[1]], dtype=int)
    h5fi["vallabels"] = np.array(vallabels, dtype=int)
    # h5fi["vallabelnums"] = np.array(vallabelnums, dtype=int)
    h5fi.close()

    # write dicts
    revvocab = dict(((v, k) for k, v in vocab.iteritems()))
    revlabels = dict(((v, k) for k, v in labeldict.iteritems()))
    with open(outfile.split('.')[0] + ".dict", "w+") as f:
        for i in xrange(1, len(revvocab) + 1):
            f.write("%s %d \n" % (revvocab[i], i))

    with open(outfile.split('.')[0] + ".labels", "w+") as f:
        for i in xrange(1, len(revlabels) + 1):
            f.write("%s %d \n" % (revlabels[i], i))


def mask_output(input_path, path="rotowire"):
    with open(os.path.join(path, "train.json"), "r") as f:
        trdata = json.load(f)

    all_ents, players, teams, cities = get_ents(trdata)
    all_ents = set([x.replace(' ', '_') for x in all_ents])

    with open(input_path, "r") as f:
        sents = f.readlines()

    masked_sents = []
    for idx, sent in enumerate(sents):
        sent = sent.split()
        ents = extract_entities(sent, all_ents, prons)
        nums = extract_numbers(sent)
        ranges = []
        for ent in ents:
            ranges.append((ent[0], ent[1], 'ENT'))
        for num in nums:
            ranges.append((num[0], num[1], 'NUM'))
        ranges.sort(key=lambda x: x[0])

        masked_sent = []
        i = 0
        while i < len(sent):
            match = False
            for r in ranges:
                if i == r[0]:
                    match = True
                    masked_sent.append(r[2])
                    i = r[1]
                    break
            if not match:
                masked_sent.append(sent[i])
                i += 1
        masked_sents.append(masked_sent)

    with open(input_path + '.masked', 'w') as f:
        f.write('\n'.join([' '.join(s) for s in masked_sents]))


def save_ent(output_path, path="rotowire"):
    with open(os.path.join(path, "train.json"), "r") as f:
        trdata = json.load(f)

    all_ents, players, teams, cities = get_ents(trdata)
    all_ents = set([x.replace(' ', '_') for x in all_ents])

    with open(output_path, 'w') as f:
        json.dump(list(all_ents), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility Functions')
    parser.add_argument('-input_path', type=str, default="",
                        help="path to input")
    parser.add_argument('-output_fi', type=str, default="",
                        help="desired path to output file")
    parser.add_argument('-gen_fi', type=str, default="",
                        help="path to file containing generated summaries")
    parser.add_argument('-dict_pfx', type=str, default="roto-ie",
                        help="prefix of .dict and .labels files")
    parser.add_argument('-val_file', type=str, default=os.path.join("nba_data", "gold.valid.txt"),
                        help="file as reference in prep_gen_data mode, of which every entry is in the form entry|attribute|value")
    parser.add_argument('-mode', type=str, default='ptrs',
                        choices=['ptrs', 'make_ie_data', 'prep_gen_data', 'extract_sent', 'mask', 'save_ent'],
                        help="what utility function to run")

    args = parser.parse_args()

    if args.mode == 'ptrs':
        make_pointerfi(args.output_fi, inp_file=args.input_path)
    elif args.mode == 'make_ie_data':
        save_full_sent_data(args.output_fi, path=args.input_path, multilabel_train=True)
    elif args.mode == 'prep_gen_data':
        prep_generated_data(args.gen_fi, args.dict_pfx, args.output_fi,
                            train_file=os.path.join(args.input_path, "train.json"),
                            val_file=args.val_file)
    elif args.mode == 'extract_sent':
        extract_sentence_data(args.output_fi, path=args.input_path)
    elif args.mode == 'mask':
        mask_output(args.input_path)
    elif args.mode == 'save_ent':
        save_ent(args.output_fi)
