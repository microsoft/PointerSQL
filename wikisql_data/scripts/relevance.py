from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import svm
import numpy as np
import random
import operator


np.random.seed(123)

def load_file(filename):
  data = []
  with open(filename) as f:
    for idx, line in enumerate(f):
      line = line.rstrip()
      if idx % 4 == 0:
        col_name = line
      elif idx % 4 == 1:
        question = line
      elif idx % 4 == 2:
        sql = line
      elif idx % 4 == 3:
        data.append((col_name, question, sql))
    return data


def question_classifier(data):
  questions = [i[1] for i in data]
  sql_type = [i[2].split(' ')[1] for i in data]
  sql_type_set = set(sql_type)
  sql_classes = dict([(type, i) for i, type in enumerate(sql_type_set)])

  target = np.array([sql_classes[i] for i in sql_type])
  sql_type_to_indices = {}
  for type in sql_type_set:
    sql_type_to_indices[type] = [idx for idx, i in enumerate(sql_type) if i == type]

  # Build classifier
  # TODO better ones 
  text_clf = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', svm.LinearSVC())])
  text_clf.fit(questions, target)
  predicted = text_clf.predict(questions)
  print('Training Acc.: %f' %(np.mean(predicted == target)))

  return sql_type_to_indices, text_clf


def find_support_sets_rank(data, ref_data, ref_clf, ref_sql_type_to_indices):
  # Rank similarity based on similar len of questions, same table, same type
  questions = [i[1] for i in data]
  ref_question_lens = [len(i[1].split(' ')) for i in ref_data]
  ref_question_len_idx_dict = dict((qlen, [i for i in range(len(ref_question_lens))
                                           if ref_question_lens[i] == qlen]) for qlen in
                                   range(min(ref_question_lens), max(ref_question_lens)+1))

  table_ids = [i[0].split(' ')[0] for i in data]
  ref_tables_ids = [i[0].split(' ')[0] for i in ref_data]
  ref_table_id_to_index_dict = dict([(table_id, []) for table_id in  ref_tables_ids])
  for i, table_id in enumerate(ref_tables_ids):
    ref_table_id_to_index_dict[table_id].append(i)

  sql_type = [i[2].split(' ')[1] for i in data]
  sql_classes = {'count': 0, 'min': 1, 'max': 2, 'sum': 3, 'avg': 4, 'select': 5}
  sql_classes_ids = {0: 'count', 1: 'min', 2: 'max', 3: 'sum',  4: 'avg',  5: 'select'}
  target = np.array([sql_classes[i] for i in sql_type])

  predicted = ref_clf.predict(questions)
  print('Acc.: %f' %(np.mean(predicted == target)))

  # Output related indices
  support_idx = []
  num_choice = 5
  for idx, pred in enumerate(predicted.tolist()):
    candidate_set = set(ref_sql_type_to_indices[sql_classes_ids[pred]])
    if data == ref_data and idx in candidate_set:  ## ignore itself
      candidate_set.remove(idx)
    id_to_score_dict = dict([(i, 0) for i in candidate_set])

    question_len = len(questions[idx].split(' '))
    # similiar len (+-2) +=1
    for ilen in range(max(question_len - 4, min(ref_question_lens)), min(max(ref_question_lens), question_len + 5)):
      ref_idx_list = list(candidate_set.intersection(set(ref_question_len_idx_dict[ilen])))
      random.shuffle(ref_idx_list)
      for ref_idx in ref_idx_list:
          if abs(ilen - question_len) <= 2:
            id_to_score_dict[ref_idx] += 0.5

    # top scores ones + random selected ones
    _top_candidates = [i[0] for i in sorted(id_to_score_dict.items(), key=operator.itemgetter(1), reverse=True) if i[1] >= 1.5]
    random.shuffle(_top_candidates)
    _top_candidates = _top_candidates[:num_choice]

    _mid_candidates = [i[0] for i in sorted(id_to_score_dict.items(), key=operator.itemgetter(1),
                                                   reverse=True) if i[1] > 0 and i[1] < 1.5]
    random.shuffle(_mid_candidates)
    _mid_candidates = _mid_candidates[:num_choice - len(_top_candidates)]

    _random_candidate = [random.choice(list(candidate_set)) for _ in range(num_choice - len(_top_candidates) - len(_mid_candidates))]
    random.shuffle(_random_candidate)

    support_idx.append(_top_candidates + _mid_candidates + _random_candidate)
    print(idx, table_ids[idx], support_idx[-1])

  return support_idx


def print_support_set(support_idx, data, filename):
  num_choice = len(support_idx[0])
  for i in range(num_choice):
    with open(filename + '_%d' % i, 'w') as f:
      for support in support_idx:
        col_name, question, sql = data[support[i]]
        f.write('%s\n%s\n%s\n\n' %(col_name, question, sql))


if __name__ == '__main__':
  filedir  = 'input/data/'
  train_filename =  filedir + 'wikisql_train.dat'
  dev_filename = filedir + 'wikisql_dev.dat'
  test_filename = filedir + 'wikisql_test.dat'
  train_data = load_file(train_filename)
  dev_data = load_file(dev_filename)
  test_data = load_file(test_filename)

  sql_type_to_indices, classifier = question_classifier(train_data)

  output_filedir = 'nl2prog_input/nl2prog_input_support_rank/'

  # Based on classification + heuristic ranking + use sql length first
  train_support_idx = find_support_sets_rank(train_data, train_data, classifier, sql_type_to_indices)
  print_support_set(train_support_idx, train_data, output_filedir + 'wikisql_train_support_rank')
  dev_support_idx = find_support_sets_rank(dev_data, train_data, classifier, sql_type_to_indices)
  print_support_set(dev_support_idx, train_data, output_filedir + 'wikisql_dev_support_rank')
  test_support_idx = find_support_sets_rank(test_data, train_data, classifier, sql_type_to_indices)
  print_support_set(test_support_idx, train_data, output_filedir + 'wikisql_test_support_rank')


