# -*- coding: utf-8 -*-

from __future__ import print_function

import json
import os
from functools import reduce

agg_ops = ['select', 'max', 'min', 'count', 'sum', 'avg']
cond_ops = ['=', '>', '<', 'OP']


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def normalize_case(s):
    s = s.replace("`", "\'").lower()
    return s


def normalize_phrase(s):
    return "^".join(s.split())


def load_tables(file):
    tables = {}
    with open(file, "r") as f:
        for l in f.readlines():
            l = normalize_case(l)
            raw = json.loads(l)

            raw['header'] = [normalize_phrase(h) for h in raw["header"]]
            types = raw['types']
            rows = raw['rows']
            for i in range(len(rows)):
                rows[i] = [normalize_phrase(t) if not is_number(t) else t for j, t in enumerate(rows[i])]

            tables[raw["id"]] = raw

    return tables


def load_data(file, orig_tables, annotated_tables, add_wrong_entry=False):
    dataset = []
    normalized_orig_dataset = []

    with open(file, "r") as f:
        lines = f.readlines()
        print("num_entry in {} : {}".format(file, len(lines)))
        for l in lines:            
            l = normalize_case(l)
            entry = json.loads(l)

            table_id = entry["table_id"]

            orig_table = orig_tables[table_id]
            annotated_table = annotated_tables[table_id]

            orig_q = entry["question"].strip()
            annotated_q = entry["annotated_question"]

            sql_components = [table_id, 
                              agg_ops[entry["sql"]["agg"]], 
                              orig_table['header'][entry["sql"]["sel"]]]

            # each element is a pair of (annotated_const, orig_const)
            # we want to match with the annotated one and replace with the orig one
            all_consts = []

            flag_query_wrong = False
            flag_nl_missing_const = False

            # processing query
            for cond in entry['sql']['annotated_conds']:
                col = orig_table['header'][cond[0]]
                op = cond_ops[cond[1]]

                if not is_number(cond[2]):

                    const = normalize_phrase(cond[2])
                    const = normalize_const_to_table_entry(const, orig_table,
                                                           annotated_table)

                    all_consts.append((cond[2], const))
                    if const is None:
                        # encounter error
                        # the constant in the query does not appear
                        # in anywhere in the table
                        print("<<<")
                        print(cond[2])
                        print(const)
                        print(l)
                        print(">>>")
                        flag_query_wrong = True
                else:
                    # it's a number, so its normalized form is also a number
                    const = cond[2]

                all_consts.append((cond[2], const))
                sql_components.extend([col, op, str(const)])

            if flag_query_wrong:
                if add_wrong_entry:
                    # add a placeholder for wrong query
                    dataset.append((table_id + " " + " ".join(orig_table["header"]) + " <ERR>", processed_q, "<ERR>"))
                # we are not adding them into the dataset
                continue

            # processing question
            processed_q = identify_phrases(annotated_q, all_consts)

            all_headers = []
            for i, annotated_h in enumerate(annotated_table["header"]):
                orig_h = orig_table["header"][i]
                all_headers.append((annotated_h, orig_h))

            processed_q = identify_phrases(processed_q, all_headers, False)

            # annotate column name
            head_to_values = {}
            for i, h in enumerate(orig_table["header"]):
                vals = [r[i] for r in orig_table["rows"]]
                head_to_values[h] = vals

            for p in all_consts:
                related_header = []
                for h in head_to_values:
                    for v in head_to_values[h]:
                        if v == p[1]:
                            related_header.append(h)
                            break

                if len(related_header) == 0:
                    pass
                elif len(related_header) == 1 and related_header[0] not in processed_q:
                    processed_q = processed_q.replace(" " + p[1] + " ", " " + related_header[0] + " " + p[1] + " ")
                else:
                    none_in_list = True
                    for h in related_header:
                        if h in processed_q:
                            none_in_list = False
                    if none_in_list:
                        pass

            # check if there is any missing constants
            miss_const = False
            missed_const = []
            for p in all_consts:
                if p[1] not in processed_q.split():
                    miss_const = True
                    missed_const.append((p[0],p[1]))

            if miss_const:
                print("# [Warning] the following question will be discarded " 
                    + " due to not containing question contants.")
                print(sql_components)
                print(missed_const)
                print(annotated_q)
                print(processed_q)
                print("")
                # we are not adding them into the dataset

                if add_wrong_entry:
                    # add a placeholder for wrong query
                    dataset.append((table_id + " " + " ".join(orig_table["header"]) + " <ERR>", 
                                    processed_q, "<ERR>"))
                continue

            sql = " ".join(sql_components)

            dataset.append((table_id + " " + " ".join(orig_table["header"]), processed_q, sql))

    print("# dataset size: {}".format(len(dataset)))
    return dataset


def normal_equal(tokens1, tokens2):
    # equivalence between strings without considering
    s1 = "".join(tokens1).replace("^","").replace("\xa0","")
    s2 = "".join(tokens2).replace("^","").replace("\xa0","")

    ss1 = s1.replace("—", "-").replace("–","-").replace("--","-")
    ss2 = s2.replace("—", "-").replace("–","-").replace("--","-")

    return ss1 == ss2

def identify_phrases(q, candidates, consider_permutation=True):
    # processing question using candidates tokens
    q = q.lower().replace("\xa0"," ")
    for annotated_v, orig_v in candidates:
        # just in case...
        annotated_v = annotated_v.lower()

        if annotated_v in q.split():
            q = q.replace(annotated_v, orig_v)
        elif orig_v in q.split():
            continue
        else:
            v_list = annotated_v.split()
            orig_v_list = orig_v.split("^")

            q_list = q.split()

            best_match = None
            best_match_score = -1

            for i in range(len(q_list)):
                for j in range(i, len(q_list)):
                    score_ij = overlap_score(q_list[i:j+1], v_list)  

                    if score_ij > best_match_score:
                        best_match = " ".join(q_list[i:j+1])
                        best_match_score = score_ij

                    if (best_match_score < 9999 and (normal_equal(q_list[i:j+1], v_list) 
                        or normal_equal(q_list[i:j+1], orig_v_list))):
                        best_match = " ".join(q_list[i:j+1])
                        best_match_score = 9999

                    if (consider_permutation and best_match_score <= 8888 
                        and is_permutation(q_list[i:j+1], v_list)):
                        best_match = " ".join(q_list[i:j+1])
                        max_score = 8888

            if best_match_score >= 0.5:
                q = q.replace(best_match, orig_v)
    return q


def normalize_const_to_table_entry(val, orig_table, annotated_table):
    # l_const is a constant represented in list form 
    # find a table entry to normalize the value val 
    # (orig_table is the original unprocessed table and annotated in the one after process)
    if is_number(val):
        # no need to normalize
        return val

    val = val.lower()
    val_split = val.split("^")
    best_match = None

    for i in range(len(orig_table["rows"])):
        for j in range(len(orig_table["rows"][i])):

            orig_v = str(orig_table["rows"][i][j]).lower()
            annotated_v = str(annotated_table["rows"][i][j]).lower()

            v_split = annotated_v.split("^")

            if val == annotated_v or val == orig_v:
                return orig_v

            if "".join(val_split) == "".join(v_split):
                # they match exactly but with wrong noise
                # we have not yet find the exact value
                best_match = orig_v

    return best_match


def find_best_match(val, lst):
    # find the best matching sequence of the token in lst for the string val
    best_score = 0
    best_match = None
    for v in lst:
        if overlap_score(v.split(), val) > best_score:
            best_score = overlap_score(v.split(), val)
            best_match = v
    return best_match


def is_permutation(l1, l2, ignore=[","]):
    # check if l1 is a permutation of l2
    if l1[0] in ignore or l2[0] in ignore:
        return False

    l1_bag = {}
    for v in l1:
        if v in ignore:
            continue
        if not v in l1_bag:
            l1_bag[v] = 0
        l1_bag[v] += 1
    l2_bag = {}

    for v in l2:
        if v in ignore:
            continue
        for i in ignore:
            if i in v:
                v = v.replace(",","")
        if not v in l2_bag:
            l2_bag[v] = 0
        l2_bag[v] += 1
    
    if len(l1_bag) != len(l2_bag):
        return False

    for v in l1_bag:
        if v not in l2_bag:
            return False
        if l2_bag[v] != l1_bag[v]:
            return False

    return True

def is_subseq(l1, l2):
    l1 = reduce((lambda x, y: x + y), [x.split(".") for x in l1])
    l2 = reduce((lambda x, y: x + y), [x.split(".") for x in l2])
    i = 0
    for j in range(len(l2)):
        if l2[j] == l1[i]:
            i += 1
        if i >= len(l1):
            return True
    return False

def overlap_score(l1, l2):
    overlap_cnt = 0
    l1 = reduce((lambda x, y: x + y), [x.split(".") for x in l1])
    l2 = reduce((lambda x, y: x + y), [x.split(".") for x in l2])
    for t in l2:
        if t in l1:
            overlap_cnt += 1
    return float(overlap_cnt) / float(len(l1) + len(l2))

if __name__ == '__main__':
    data_dir = "data"
    out_dir = "."

    datasets = ["train", "dev", "test"] 
    for dataset in datasets:
        print("# processing dataset {}".format(dataset))
        orig_tables = load_tables(os.path.join(data_dir, dataset + ".tables.jsonl"))
        annotated_tables = load_tables(os.path.join(data_dir, dataset + ".tables.annotated.jsonl"))
        result = load_data(os.path.join(data_dir, dataset + ".annotated.jsonl"), 
                           orig_tables, annotated_tables, True)

        with open(os.path.join(out_dir, "wikisql_{}.dat".format(dataset)), "w") as f:
            for entry in result:
                f.write(entry[0] + "\n")
                f.write(entry[1] + "\n")
                f.write(entry[2] + "\n")
                f.write("\n")
