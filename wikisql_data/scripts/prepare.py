import os
import sys
import json
from pprint import pprint
from functools import reduce

USE_COLUMN_ANNOTATION = False

def prepare_data(question_file, table_file):
    with open(question_file, "r") as qf, open(table_file) as tf:
        for l in qf.readlines():
            question = json.loads(l)
            print(question)
            break

def strip_brackets(s):
    return s.replace("-lrb-", "(").replace("-rrb-", ")")

def load_annotated_tables(file):
    tables = {}
    with open(file, "r") as f:
        for l in f.readlines():
            raw = json.loads(l.replace("`","\'"))
            tables[raw["id"]] = raw
    return tables

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def normalize_const_to_table_entry(val, head_to_values, exact_match=False):
    # l_const is a constant represented in list form 
    if is_number("".join(val)):
        # no need to normalize
        return val

    best_score = 0
    best_match = None
    for h in head_to_values:
        for v in head_to_values[h]:
            if not exact_match and overlap_score(v.split("^"), val) > best_score:
                best_score = overlap_score(v.split("^"), val)
                best_match = v

            if "".join(val) == v or "".join(val) == "".join(v.split("^")) or "".join(val).replace("`","\'") == "".join(v.split("^")):
                # they match exactly but with wrong noise
                best_score = 999
                best_match = v.split("^")

    return best_match

def find_best_match(val, lst):
    best_score = 0
    best_match = None
    for v in lst:
        if overlap_score(v.split(), val) > best_score:
            best_score = overlap_score(v.split(), val)
            best_match = v
    return best_match

def is_permutation(l1, l2, ignore=[","]):

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

def load_annotated_data(file, tables=None, process_level=2):
    """ load annotated examples from the datafile """
    result = []
    valid_cnt = 0
    excluded_cases = []
    error_cases = []
    wrong = 0

    with open(file, "r") as f:
        for l in f.readlines():
            raw = json.loads(l)

            head_dict = {}
            for h in raw["table"]["header"]:
                h_str = ""
                for i in range(len(h["words"])):
                    h_str += h["words"][i] + (h["after"][i] if h["after"][i] is not " " else "^")
                head_dict[" ".join(h["words"])] = h_str
            
            header = " ".join([head_dict[k] for k in head_dict])
            
            # maintain mappings from table header to its values, used to manually set up head names
            if tables:
                head_to_values = {}
                for row in tables[raw["table_id"]]["content"]:
                    for i in range(len(row)):
                        if tables[raw["table_id"]]["header"][i] not in head_to_values:
                            head_to_values[tables[raw["table_id"]]["header"][i]] = [] 
                        head_to_values[tables[raw["table_id"]]["header"][i]].append(row[i]) 

            q = raw["seq_input"]["words"][26:]
            q = " ".join(q[q.index("symquestion")+1 : q.index("symend")]).replace("`","\'")
            raw_q = q

            a = raw["seq_output"]["words"]
            select = a[a.index("symselect") + 1 : a.index("symwhere") if "symwhere" in a else a.index("symend")]
            agg = select[select.index("symagg") + 1 : select.index("symcol")]

            sel_raw = " ".join(select[select.index("symcol") + 1 : ])

            if sel_raw in head_dict:
                select_col = head_dict[sel_raw]
            else:
                select_col = head_dict[find_best_match(select[select.index("symcol") + 1 : ], [h for h in head_dict])]

            where = raw["where_output"]["words"] # a[a.index("symwhere") + 1 : a.index("symend")]
            predicates = []
            current_pred = None
            for tok in where:
                if tok == "symwhere":
                    continue
                if tok == "symcol":
                    current_pred = []
                if tok in ["symend", "symand"]:
                    if current_pred:
                        predicates.append(current_pred)
                else:
                    current_pred.append(tok)

            const_dict = {}
            used_cols = []

            all_preds = [] 
            for pred_toks in predicates:

                raw_col = " ".join(pred_toks[pred_toks.index("symcol") + 1 : pred_toks.index("symop")])
                if raw_col not in head_dict:
                    col = head_dict[find_best_match(pred_toks[pred_toks.index("symcol") + 1 : pred_toks.index("symop")], [h for h in head_dict])]
                else:
                    col = head_dict[raw_col]

                op = pred_toks[pred_toks.index("symop") + 1 : pred_toks.index("symcond")][0]
                val = pred_toks[pred_toks.index("symcond") + 1 : ]

                normalized_val = normalize_const_to_table_entry(val, head_to_values)

                if normalized_val == None:
                    wrong += 1
                else:
                    val = normalized_val

                const_dict[" ".join(val)] = "^".join(val)
                
                pred = col + " " + op + " " + "^".join(val)
                
                used_cols.append(col)
                all_preds.append(pred)

            used_cols.append(select_col)

            pred_str = " ".join(all_preds)
            output_str = (agg[0] if agg else "select") + " " + select_col + " " + pred_str
            
            if process_level >= 1:

                # replace unconsecutive phrases into a concatated phase 
                #   that allows pointer to points pointed to
                # e.g. date time --> date^time if date^time is a constant in the table
                # replace the highest score term based on overlapping
                for k in const_dict:

                    if k in q.split():
                        q = q.replace(k, const_dict[k])
                    else:
                        if const_dict[k] in q.split():
                            continue

                        q_list = q.split()

                        max_score = 0
                        to_subst = None
                        for i in range(len(q_list)):
                            for j in range(i + 1, len(q_list)):

                                if max_score < 9999 and "".join(q_list[i:j+1]) == "".join(k.replace("^","").replace(" ","")):
                                    to_subst = " ".join(q_list[i:j+1])
                                    max_score = 9999

                                if max_score == 0 and is_permutation(q_list[i:j+1], k.replace("^"," ").split()):
                                    to_subst = " ".join(q_list[i:j+1])
                                    max_score = 8888
                        
                        if to_subst:
                            q = q.replace(to_subst, const_dict[k])

            if process_level >= 1:

                # replace unconsecutive phrases into a concatated phase thta can be directly pointed to
                # e.g. date time --> date^time if date^time is a header name
                # only replace when it is full match since this is header (not mandatary)

                for k in head_dict:

                    if k in q.split():
                        q = q.replace(k, head_dict[k])
                    else:

                        if head_dict[k] in q.split():
                            continue

                        k_list = k.split()
                        q_list = q.split()

                        # deal with cases with plural cases, 
                        # this is only used on processing level 2 or above
                        if process_level >=2 and len(k_list) == 1:
                            for i in range(len(q_list)):
                                if len(k_list[0]) > 2 and edit_distance(q_list[i], k_list[0]) == 1:
                                    q = q.replace(q_list[i], head_dict[k])
                            continue

                        best_match = None
                        best_match_score = -1

                        for i in range(len(q_list)):
                            for j in range(i, len(q_list)):
                                score_ij = overlap_score(q_list[i:j+1], k_list)  

                                if score_ij > best_match_score:
                                    best_match = " ".join(q_list[i:j+1])
                                    best_match_score = score_ij

                        if best_match_score >= 0.5:
                            q = q.replace(best_match, head_dict[k])
            
            if process_level >= 2 and tables and USE_COLUMN_ANNOTATION:
                """ 
                Process questions so that constants appearing in the table 
                have their column names associated with them
                """
                for k in const_dict:
                    related_header = []

                    for h in head_to_values:
                        for v in head_to_values[h]:
                            if v.replace("^","") == const_dict[k].replace("^", ""):
                                related_header.append(h)
                                break

                    if len(related_header) == 0:
                        pass
                    elif len(related_header) == 1 and related_header[0] not in q:
                        q = q.replace(" " + const_dict[k] + " ", " " + related_header[0] + " " + const_dict[k] + " ")
                    else:
                        none_in_list = True
                        for h in related_header:
                            if h in q:
                                none_in_list = False
                        if none_in_list:
                            pass

            valid = True
            for k in const_dict:
                if const_dict[k] not in q.split():
                    valid = False
            #for c in used_cols:
            #    if c not in q.split():
            #        valid = False

            if valid:
                valid_cnt += 1
                print(strip_brackets(raw["table_id"] + " " + header))
                print(strip_brackets(q))
                print(strip_brackets(raw["table_id"] + " " + output_str))
                print("")
            else:
                #print("-->")
                #print(raw_q)
                #print(q)
                #print(output_str)
                #print("")
                #print(used_cols)
                #print([const_dict[k] for k in const_dict])
                #pprint(head_dict)
                #print("")
                excluded_cases.append((raw["table_id"] + " " + header, q, raw["table_id"] + " " + output_str))
    
    print("dudulu")
    print(len(excluded_cases))
    print(valid_cnt)
    for e in excluded_cases:
        print(e[0])
        print(e[1])
        print(e[2])
        print("")


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
    

def edit_distance(s1, s2):
    m = len(s1)+1
    n = len(s2)+1
    tbl = {}
    for i in range(m): tbl[i,0] = i
    for j in range(n): tbl[0,j] = j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j - 1] + 1, tbl[i - 1, j] + 1, tbl[i - 1, j - 1] + cost)
    return tbl[i,j]

if __name__ == '__main__':
    #prepare_data("data/dev.jsonl", "data/dev.tables.jsonl")

    datasets = ["train", "dev", "test"] #["dev", "train", "test"]
    for dataset in datasets:
        tables = load_annotated_tables(os.path.join("..", "annotated", dataset) + ".tables.jsonl")
        load_annotated_data(os.path.join("..", "annotated", dataset) + ".jsonl", tables)
        print("======================================")
