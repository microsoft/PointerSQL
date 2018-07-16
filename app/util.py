import os

def _read_dataset(filename, input_num=1, output_num=1):
    dataset = []
    N = input_num + output_num + 1
    with open(filename, "r") as f:
        raw_lines = f.readlines()
        for i in range(0, len(raw_lines)):
            k = i % N
            if k >= 0 and k < input_num:
                if k == 0:
                    entry = {}
                if input_num == 1:
                    entry["in"] = raw_lines[i].strip().split()
                entry["in{}".format(k + 1)] = raw_lines[i].strip().split()

            if k >= input_num and k < input_num + output_num:
                if k == input_num and output_num == 1:
                    entry["out"] = raw_lines[i].strip().split()

                if output_num > 1:
                    entry["out{}".format(k + 1 - input_num)] = raw_lines[i].strip().split()

            if k == input_num + output_num:
                dataset.append(entry)
                continue
    return dataset

def read_dataset(filename, support_file_list=None, input_num=1, output_num=1):
    """ Read a file and prepare a dataset """
    support_dataset = None
    if support_file_list:
        support_dataset = [_read_dataset(filename=_filename, input_num=input_num,
                                        output_num=output_num) for _filename in support_file_list]

    dataset = _read_dataset(filename=filename, input_num=input_num, output_num=output_num)

    if support_file_list:
        return dataset, support_dataset
    else:
        return dataset


def find_file(root_dir, file_name):
    """ given a root_dir, find the file inside the dir with the file named file_name"""
    candidates = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == file_name:
                 candidates.append(os.path.join(root, file))
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) == 0:
        print("[ERROR] File {} is not find inside root dir {}".format(file_name, root_dir))
        return None
    else:
        print("[ERROR] Multiple files named {} are found inside root dir {}".format(file_name, root_dir))
        return None
