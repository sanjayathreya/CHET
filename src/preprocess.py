import pandas as pd
from pyhealth.medcode import CrossMap
from pyhealth.datasets import MIMIC4Dataset,MIMIC3Dataset
from pyhealth.datasets.utils import flatten_list
from pyhealth.medcode import InnerMap
import os
import pickle
import numpy as np
from collections import OrderedDict
from pyhealth.tokenizer import Tokenizer
from tqdm import tqdm
from utils import format_time
import time

icd9cm = InnerMap.load("ICD9CM")

def create_parsed_datasets(patient_dict, tablename):
  """Parser function to create base dictionaries

  This function uses all patients from Pyhealth dataset along
  with diagnosis_icd for each admission to create
  paitient-admission and admission-diagnosis codes

  Parameters
  ----------
  patient_dict : dict
      paitient admission or visit dictionary.
  patient_dict : dict
    paitient admission or visit dictionary.
  Returns
  -------
  dict, dict
      paitient to list of admissions.
      admissions to list of diagnosis codes.
  """
  del_pid = {}
  patient_admission = OrderedDict()
  admission_codes = OrderedDict()

  for pid, values in tqdm(patient_dict.items(), desc="Creating Parsed datasets for each paitient"):
    patient = patient_dict[pid]
    visit_dict = patient.visits
    # we parse patients who have greater than 2 visits
    if (len(visit_dict) >= 2):
      admissions = []
      for visit_key, visit_values in visit_dict.items():
        diagnoses = visit_values.get_code_list(table=tablename)
        diagnoses_std = []
        for code in diagnoses:
          std = icd9cm.standardize(code)
          if std in icd9cm:
            diagnoses_std.append(std)
          elif std[:1] not in ['E', 'V', 'N'] and str(float(std)) in icd9cm: #exclude No.DX
            diagnoses_std.append(str(float(std)))
            # print(f'\n code{code} using float map')
          else:
            diagnoses_std.append('NoDx')
            # print(f'\n code{code} has NO MAP')
        admissions.append({'adm_id': visit_key, 'adm_time': visit_values.encounter_time})
        admission_codes[visit_key] = diagnoses_std

        # if there is a diagnose code with no mapping then drop the patient and
        counter = 0
        counter = sum([counter + 1 for diagnoses in diagnoses_std if diagnoses == '' or diagnoses == 'NoDx'])
        if (len(diagnoses) == 0 or counter != 0):
          del_pid[pid] = pid
      patient_admission[pid] = sorted(admissions, key=lambda admission: admission['adm_time'])

  for pid in del_pid.keys():
    patient = patient_dict[pid]
    visit_dict = patient.visits
    del patient_admission[pid]
    for visit_key, visit_values in visit_dict.items():
      del admission_codes[visit_key]

  return patient_admission, admission_codes

def generate_samples( patient_admission, admission_codes, seed, sample_size=10000):
  """Parser function to generate samples

  This function uses population of  paitient-admission
  and admission-diagnosis codes for a sampling process

  Parameters
  ----------
  patient_admission : dict
      paitient admission or visit dictionary.
  admission_codes : dict
    admission to diagnosis codes dictionary
  seed : int
    random number seed
  Returns
  -------
  dict, dict
      paitient to list of admissions.
      admissions to list of diagnosis codes.
  """
  np.random.seed(seed)
  keys = list(patient_admission.keys())
  selected_pids = np.random.choice(keys, sample_size, False)
  patient_admission_sample = {pid: patient_admission[pid] for pid in selected_pids}
  admission_codes_sample = dict()
  for admissions in patient_admission_sample.values():
      for admission in admissions:
          adm_id = admission['adm_id']
          admission_codes_sample[adm_id] = admission_codes[adm_id]
  return patient_admission_sample, admission_codes_sample


def get_stats( patient_admission, admission_codes):
  """Helper function to return stats.

  Function that returns stats of parsed datasets

  Parameters
  ----------
  patient_admission : dict
      patient to admission dictionary.
  admission_codes : dict
      admission to diagnosis codes dictionary.
  Returns
  -------
  int
      max_admission_num - maximum number of admissions.
  """
  patient_num = len(patient_admission)
  max_admission_num = max([len(admissions) for admissions in patient_admission.values()])
  avg_admission_num = sum([len(admissions) for admissions in patient_admission.values()]) / patient_num
  max_visit_code_num = max([len(codes) for codes in admission_codes.values()])
  avg_visit_code_num = sum([len(codes) for codes in admission_codes.values()]) / len(admission_codes)
  print('patient num: %d' % patient_num)
  print('max admission num: %d' % max_admission_num)
  print('mean admission num: %.2f' % avg_admission_num)
  print('max code num in an admission: %d' % max_visit_code_num)
  print('mean code num in an admission: %.2f' % avg_visit_code_num)

  return max_admission_num

def save_sparse(path, x):
  idx = np.where(x > 0)
  values = x[idx]
  np.savez(path, idx=idx, values=values, shape=x.shape)

def save_files(path, type="", **kwargs):
  """Helper to save files.

  Helper function that takes in a path and
  saves the files passed in by key value pairs
  to the given path

  Parameters
  ----------
  path : string
      path of given file.
  kwargs : dict
    key value pairs of file name and file content.
  Returns
  -------
  None
      Returns nothing.
  """
  if not os.path.exists(path):
    os.makedirs(path)

  if type !='standard':
    for key, value in kwargs.items():
      name = key+'.pkl'
      pickle.dump(value, open(os.path.join(path, name), 'wb'))
      print(f'saved {key} data ...')

  elif  type =='standard':
    for key, value in kwargs.items():
      if key in ['code_x',  'code_y',  'divided', 'neighbors', 'code_adj']:
        save_sparse(os.path.join(path, key), value)
      elif key =='visit_lens':
        np.savez(os.path.join(path, key), lens = value)
      elif key == 'hf_y':
        np.savez(os.path.join(path, key), hf_y = value)

  return None

def get_levels_dict(level_dict):
  """helper to get a specific levels dictionary

  specifies the levels hierarchy needed

  Parameters
  ----------
  level_dict
        A specific levels dictionary.

  Returns
  -------
  dict
      mapping of levels to hierarchy.
  """
  counter = 0
  for k, v in level_dict.items():
    level_dict[k] = counter
    counter += 1
  return level_dict


def get_levels_dicts():
  """get dictionaries of diagnosis levels

  function uses the graph network in pyhealth icd9cm mapping
  to discern a 3 level anscestoral hierarchy. Similar to the
  implementation by authors certain levels are forced into the
  3 level hierarchy

  Parameters
  ----------
  None

  Returns
  -------
  None
  """
  codes = list(icd9cm.graph.nodes.keys())
  level3 = dict()
  hierarchy = []
  for code in codes:
    if "-" not in code:
      dignoses_code = code.split('.')[0]
      if dignoses_code not in level3:
        level3[dignoses_code] = len(level3)
    else:
      hierarchy.append(code)

  single_name_codes1 = [str(code) for code in range(280, 290)]
  single_name_codes2 = [str(code) for code in range(740, 760)]
  hierarchy = hierarchy + single_name_codes1 + single_name_codes2

  levels_map = {}
  for code in hierarchy:
    if code not in ['001-999.99', '290-299.99', '800-829.99', '870–897.99']:
      ancestors = icd9cm.get_ancestors(code)
      if '001-999.99' in ancestors and len(ancestors) > 2:
        levels_map[code] = len(ancestors) - 2
      elif '001-999.99' in ancestors or (code[:1] == 'E' and len(ancestors) == 2):
        levels_map[code] = len(ancestors) - 1
      else:
        levels_map[code] = len(ancestors)

  level1 = OrderedDict(sorted({k: v for (k, v) in levels_map.items() if v == 0}.items()))
  level2 = OrderedDict(sorted({k: v for (k, v) in levels_map.items() if v == 1}.items()))

  level1 = get_levels_dict(level1)
  level2 = get_levels_dict(level2)

  return level1, level2, level3


def generate_code_levels(code_map):
  """generate diagnosis code level matrix

  Takes in a code mapping to generate the Global relationships between
  diagnoses codes.

  Parameters
  ----------
  code_map : dict
      code to index mapping.

  Returns
  -------
  np array
      [l1 idx l2 idx l3 idx code idx].
  """
  level1, level2, level3 = get_levels_dicts()
  code_level_matrix = np.zeros((len(code_map), 4), dtype=int)
  for code, cid in tqdm(code_map.items(), desc="generating code level matrix for codes"):
    # print(f'code:{code} cid {cid}')
    ancestors = icd9cm.get_ancestors(code)
    for ancestor in ancestors:
      if ancestor in level1:
        l1code = level1[ancestor]
      if ancestor in level2:
        l2code = level2[ancestor]
    l3code = level3[code.split('.')[0]]
    code_level_matrix[cid] = np.array([l1code, l2code, l3code, cid])

  return code_level_matrix

def split_patients(patient_admission, admission_codes, code_map, ratios = (.80,.10,.10), seed=6669):
  """Split datasets into train test validation.

  Takes in parsed dictionary of patient_admissions, admission_codes and
  diagnosis code to index map (code_map) and returns split datasets
  This function does not do a naive split , instead it tries to ensure most of
  diagnosis codes are seen during training phase

  Parameters
  ----------
  patient_admission : dict
      patient_admissions dictionary.
  admission_codes : dict
      admission_codes dictionary.
  code_map: dict
      code to index map dictionary.

  Returns
  -------
  list, list, list
      Returns 3 lists of train, validation and test patient ids.
  """
  np.random.seed(seed)
  common_pids = set()
  for i, code in enumerate(tqdm(code_map, desc="Split Paitents")):
    for pid, admissions in patient_admission.items():
      for admission in admissions:
        codes = admission_codes[admission['adm_id']]
        if code in codes:
          common_pids.add(pid)
          break
        else:
          continue
      break

  max_admission_num = 0
  pid_max_admission_num = 0
  for pid, admissions in patient_admission.items():
    if len(admissions) > max_admission_num:
      max_admission_num = len(admissions)
      pid_max_admission_num = pid
  common_pids.add(pid_max_admission_num)
  remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))
  np.random.shuffle(remaining_pids)

  train_num = int(len(patient_admission) * ratios[0])
  test_num = int(len(patient_admission) * ratios[2])
  valid_num = len(patient_admission) - train_num - test_num

  train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
  valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
  test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
  print(f'Split patient_admission #: {len(patient_admission)} into \n\t Train #: {len(train_pids)} \n\t Validation #: {len(valid_pids)} \n\t Test #: {len(test_pids)}')
  return train_pids, valid_pids, test_pids

def normalize_adj(adj):
  """Normalize adjacency matrix.

  Function to normalize the adjacency matrix

  Parameters
  ----------
  adj : np array
      adjacency matrix


  Returns
  -------
  np array
      normalized adjacency matrix
  """
  s = adj.sum(axis=-1, keepdims=True)
  s[s == 0] = 1
  result = adj / s
  return result

def generate_code_code_adjacent(pids, patient_admission, admission_codes_encoded, code_num, threshold=0.01):
  """generate a global code code adjacency matrix.

  Function to normalize the adjacency matrix

  Parameters
  ----------
  pids : list
      list of training patient ids
  patient_admission : dictionary
      patient and list of admissions
  admission_codes_encoded : dictionary
      admission codes encoded  dictionary
  code_num : int
      total number of diagnosis codes
  threshold : float
      Frequency threshold
  Returns
  -------
  np array
      global diagnosis code adjacency matrix
  """
  n = code_num
  adj = np.zeros((n, n), dtype=int)
  for i, pid in enumerate(tqdm(pids, desc="generating global code code adjacent matrix")):
    for admission in patient_admission[pid]:
      codes = admission_codes_encoded[admission['adm_id']]
      for i, c_i in enumerate(codes):
        for j, c_j in enumerate(codes):
          if i >= j:
            continue
          adj[c_i, c_j] += 1
          adj[c_j, c_i] += 1
  norm_adj = normalize_adj(adj)
  a = norm_adj < threshold
  b = adj.sum(axis=-1, keepdims=True) > (1 / threshold)
  adj[np.logical_and(a, b)] = 0
  return adj


def construct_x_y(pids, patient_admission, admission_codes_encoded, max_admission_num, code_num):
  """construct_x, y matrices.

  Function build the input matrix x and label matrix y
  The last visit is used as labels y the training input x consists of the rest of visits

  Parameters
  ----------
  pids : list
      list of training patient ids
  patient_admission : dictionary
      patient and list of admissions
  admission_codes_encoded : dictionary
      admission codes encoded  dictionary
  max_admission_num : int
      maximum number of admissions per
  code_num : int
      number of diagnosis codes
  Returns
  -------
  np array, np array, np array
      x matrix.
      y matrix.
      number of visits in input x for each patient
  """

  n = len(pids)
  x = np.zeros((n, max_admission_num, code_num), dtype=bool)
  y = np.zeros((n, code_num), dtype=int)
  lens = np.zeros((n,), dtype=int)
  for i, pid in enumerate(tqdm(pids, desc="building x, y matrices")):
    admissions = patient_admission[pid]
    for k, admission in enumerate(admissions[:-1]): # exclude last visit for input matrix x
      codes = admission_codes_encoded[admission['adm_id']]
      x[i, k, codes] = 1
    codes = np.array(admission_codes_encoded[admissions[-1]['adm_id']])  # last visit for labels y
    y[i, codes] = 1
    lens[i] = len(admissions) - 1 # number of visits in x
  return x, y, lens


def construct_hf_label_y(codes_y, code_map, hf_prefix='428'):
  """construct_hf_label_y.

  Function build the label matrix for heart failure
  code '428' signifies heart failure , we use the code to filter the labels

  Parameters
  ----------
  codes_y : np array
      list of training patient ids
  code_map : dictionary
      mapping of codes to indices
  hf_prefix : string
      hf code
  Returns
  -------
  np array,
      lablels for heart failyre
  """
  hf_list = np.array([cid for code, cid in code_map.items() if code.startswith(hf_prefix)])
  hfs = np.zeros((len(code_map),), dtype=int)
  hfs[hf_list] = 1
  hf_exist = np.logical_and(codes_y, hfs)
  y = (np.sum(hf_exist, axis=-1) > 0).astype(int)
  return y

def build_neighbor_codes(code_x, lens, adj):
  """construct neighbors matrix n_t.

  Function build the neighbor codes

  Parameters
  ----------
  code_x : np array
      X matrix of codes
  lens : np array
      number of visits per patient
  adj : np array
      global code-code adjacency matrix
  Returns
  -------
  np array,
      neighbors code matrices [ paitient, visit, neighbors ]
  """
  n = len(code_x)
  neighbors = np.zeros_like(code_x, dtype=bool)
  for i, admissions in enumerate(tqdm(code_x, desc="generating neighbors matrix")):
    for j in range(lens[i]):
      codes_set = set(np.where(admissions[j] == 1)[0])
      all_neighbors = set()
      for code in codes_set:
        code_neighbors = set(np.where(adj[code] > 0)[0]).difference(codes_set)
        all_neighbors.update(code_neighbors)
      if len(all_neighbors) > 0:
        neighbors[i, j, np.array(list(all_neighbors))] = 1
  return neighbors


def divide_disease_type_matrices(code_x, neighbors, lens):
  """build the transition matrices neighbors matrix n_t.

  Function build the neighbor codes

  Parameters
  ----------
  code_x : np array
      X matrix of codes
  lens : np array
      number of visits per patient
  neighbors : np array
      neighbors matrix
  Returns
  -------
  np array,
      divided disease type_ matrices [ paitient, visit, disease_types (m_p,m_en_,m_eu) ]
  """
  n = len(code_x)
  divided = np.zeros((*code_x.shape, 3), dtype=bool)
  for i, admissions in enumerate(tqdm(code_x, desc="generating divided transition matrices m_p,m_en,m_eu")):
    divided[i, 0, :, 0] = admissions[0]
    for j in range(1, lens[i]):
      codes_set = set(np.where(admissions[j] == 1)[0])
      m_set = set(np.where(admissions[j - 1] == 1)[0])
      n_set = set(np.where(neighbors[i][j - 1] == 1)[0])
      m_p = codes_set.intersection(m_set)
      m_en = codes_set.intersection(n_set)
      m_eu = codes_set.difference(m_set).difference(n_set)
      if len(m_p) > 0:
        divided[i, j, np.array(list(m_p)), 0] = 1
      if len(m_en) > 0:
        divided[i, j, np.array(list(m_en)), 1] = 1
      if len(m_eu) > 0:
        divided[i, j, np.array(list(m_eu)), 2] = 1
  return divided

def generate_parsed_datesets(dataset_name, parsed_main_path):
  """parse EHR datasets

  Function that uses pyhleath libraty to parse EHR datasets

  Parameters
  ----------
  dataset_name : string
      dataset type ('mimic3' or 'mimic4')
  Returns
  -------
  None
      returns nothing.
  """
  if dataset_name == 'mimic3':
    diagnoses_table = 'DIAGNOSES_ICD'
    ds = MIMIC3Dataset(
      root="data/mimic3/raw",
      tables=[diagnoses_table]
    )
  elif dataset_name == 'mimic4':
    diagnoses_table = 'diagnoses_icd'
    ds = MIMIC4Dataset(
      root="data/mimic4/raw",
      tables=[diagnoses_table],
      code_mapping={"ICD10CM": "ICD9CM"},
    )
  else:
    raise Exception('This data type is not supported, choose $mimic3$ or $mimic4$' )

  print('\nLoaded datasets in pyhealth\n')

  patient_dict = ds.patients
  patient_admission, admission_codes = create_parsed_datasets(patient_dict, diagnoses_table)
  print("\nsaving parsed data\n")
  save_files(parsed_main_path, patient_admission=patient_admission, admission_codes=admission_codes)

  return None

def preprocess(dataset_name, seed, sample_num = 1, from_cached = True):
  """preprocess EHRdataset

  Main preprocessing wrapper function that parses EHRdataset for a
  sample and seed and saves output in encoded and standard folders
  this function reads data from cached folders if available

  Parameters
  ----------
  dataset_name : string
      dataset type ('mimic3' or 'mimic4')
  seed : string
    dataset type ('mimic3' or 'mimic4')
  sample_num : string
    dataset type ('mimic3' or 'mimic4')
  dataset_name : string
    dataset type ('mimic3' or 'mimic4')
  Returns
  -------
  None
      returns nothing.
  """
  data_path = os.path.join('data')
  dataset = dataset_name
  dataset_path = os.path.join(data_path, dataset)
  parsed_sample_path = os.path.join(dataset_path, 'parsed', str(sample_num))
  parsed_main_path = os.path.join(data_path, dataset, 'parsed')
  encoded_path = os.path.join(dataset_path, 'encoded', str(sample_num))
  standard_path = os.path.join(dataset_path, 'standard', str(sample_num))

  if from_cached:
    patient_admission = pickle.load(open(os.path.join(parsed_main_path, 'patient_admission.pkl'), 'rb'))
    admission_codes = pickle.load(open(os.path.join(parsed_main_path, 'admission_codes.pkl'), 'rb'))
  else:
    generate_parsed_datesets(dataset_name, parsed_main_path)
    patient_admission = pickle.load(open(os.path.join(parsed_main_path, 'patient_admission.pkl'), 'rb'))
    admission_codes = pickle.load(open(os.path.join(parsed_main_path, 'admission_codes.pkl'), 'rb'))

  if dataset_name == 'mimic4':
    patient_admission, admission_codes = generate_samples(patient_admission, admission_codes, seed=seed, sample_size=10000)
    print("\nsample stats for mimic4\n")

  max_admission_num = get_stats(patient_admission, admission_codes)

  # Generate code map and encode admission_codes
  codes = list(admission_codes.values())
  codes = list(set(flatten_list(codes)))
  tokenizer = Tokenizer(tokens=codes)
  code_map = tokenizer.vocabulary.token2idx
  admission_codes_encoded = { admission_id: list(set(tokenizer.convert_tokens_to_indices(codes))) for admission_id, codes in admission_codes.items() }
  code_num = len(code_map)
  print('There are %d codes' % code_num)

  # diagnosis code levels
  code_levels = generate_code_levels(code_map)
  print("\ncompleted code levels")

  # Split dataset to train , validation, test 80/10/10
  train_pids, valid_pids, test_pids  = split_patients(patient_admission, admission_codes, code_map, ratios=(.80,.10,.10), seed=seed)
  all_pids = { 'train_pids': train_pids, 'valid_pids': valid_pids, 'test_pids': test_pids }

  print("\nsaving encoded data\n")
  save_files(encoded_path, patient_admission =patient_admission, codes_encoded=admission_codes_encoded,
            code_map = code_map, pids = all_pids)

  # generate global adjacency matrix
  adj_matrix = generate_code_code_adjacent(train_pids, patient_admission, admission_codes_encoded, code_num, threshold=0.01)
  norm_adj_matrix = normalize_adj(adj_matrix)
  save_dict = {'code_adj': norm_adj_matrix}
  save_files(standard_path, type='standard', **save_dict)

  # generate all datasets
  datasets = ['train', 'valid', 'test' ]
  for idx, dataset in enumerate(datasets):
    if dataset == 'train':
      pids = train_pids
    elif dataset == 'valid':
      pids = valid_pids
    else:
      pids = test_pids
    print(f'\nprocessing standard data for {dataset}\n')
    (X, Y, visit_lens) = construct_x_y(pids, patient_admission, admission_codes_encoded, max_admission_num, code_num)
    N_t = build_neighbor_codes(X, visit_lens, adj_matrix)
    M_t = divide_disease_type_matrices(X, N_t, visit_lens)
    Y_hf = construct_hf_label_y(Y, code_map)

    std_path = os.path.join(standard_path, dataset)
    save_dict = {'code_x': X, 'visit_lens': visit_lens , 'code_y': Y,
                 'hf_y': Y_hf, 'divided': M_t, 'neighbors': N_t}

    print(f'\nsaving standard data for {dataset}\n')
    save_files(std_path, type='standard', **save_dict )

  return None

if __name__ == '__main__':

    datasets = [ 'mimic3', 'mimic4']
    seeds = [6669, 1000, 1050, 2052, 3000]
    res = []
    for dataset in datasets:
      print(f'\n******Preprocess {dataset}******\n')
      st = time.time()
      parsed_main_path = os.path.join('data', dataset, 'parsed')
      generate_parsed_datesets(dataset, parsed_main_path)
      et = time.time()
      pyhealth_parsing_time = et-st
      sample_time = []
      for idx, seed in enumerate(seeds):
        st = time.time()
        preprocess(dataset, seed=seed, sample_num = idx, from_cached = True)
        et = time.time()
        cost = et-st
        sample_time.append(cost)

      result = {
        'dataset_name': dataset,
        'pyhealth_parsing_time': pyhealth_parsing_time,
        'sample_time_1': sample_time[0],
        'sample_time_2': sample_time[1],
        'sample_time_3': sample_time[2],
        'sample_time_4': sample_time[3],
        'sample_time_5': sample_time[4],
      }
      df_ = pd.DataFrame(result, index=[0])
      res.append(df_)
      print(f'\n******Completed Preprocess {dataset}******\n')

    df = pd.concat(res)

    # Write preprocessing time to output directory
    output_dir = os.path.join('out')
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    output_file = os.path.join(output_dir,'output_preprocess.csv')
    df.to_csv(output_file, index=False)