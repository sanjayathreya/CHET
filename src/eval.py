import os
import random
import time

import torch
import numpy as np
import pandas as pd

from model import Model
from utils import load_adj, EHRDataset, format_time
from metrics import f1, top_k_prec_recall, calculate_occurred
from train import historical_hot
from sklearn.metrics import f1_score, roc_auc_score

# def get_dataset_tensors(dataset):
#     code_x      = torch.from_numpy(dataset.code_x).to(device)
#     visit_lens  = torch.from_numpy(dataset.visit_lens).to(device=device, dtype=torch.long)
#     divided     = torch.from_numpy(dataset.divided).to(device)
#     y           = torch.from_numpy(dataset.y).to(device=device, dtype=torch.float32)
#     neighbors   = torch.from_numpy(dataset.neighbors).to(device)
#     return code_x, visit_lens, divided, y, neighbors

def eval_diag(model, dataset, dataset_name, task_name, train_index, historical):
    model.eval()
    labels = dataset.label()
    preds = []
    for step in range(len(dataset)):
        code_x, visit_lens, divided, y, neighbors = dataset[step]
    # code_x, visit_lens, divided, y, neighbors = get_dataset_tensors(dataset)
        output = model(code_x, divided, neighbors, visit_lens)
        pred = torch.argsort(output, dim=-1, descending=True)
        pred = pred.detach().cpu().numpy()
        preds.append(pred)
        print('\r    Evaluating step %d / %d' % (step + 1, len(dataset)), end='')
    preds = torch.vstack(preds).detach().cpu().numpy()
    f1_score = f1(labels, preds)
    prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])
    r1, r2 = calculate_occurred(historical, labels, preds, ks=[10, 20, 30, 40])
    print('\r    f1_score: %.4f --- top_k_recall: %.4f, %.4f, %.4f, %.4f  --- occurred: %.4f, %.4f, %.4f, %.4f  --- not occurred: %.4f, %.4f, %.4f, %.4f'
          % (f1_score, recall[0], recall[1], recall[2], recall[3], r1[0], r1[1], r1[2], r1[3], r2[0], r2[1], r2[2], r2[3]))

    result = {
        'dataset_name' : dataset_name,
        'task_name' : task_name,
        'train_index' : train_index,
        'f1_score': f1_score,
        'R@10': recall[0],
        'R@20': recall[1],
        'R@30': recall[2],
        'R@40': recall[3],
        'persistent@10': r1[0],
        'persistent@20': r1[1],
        'persistent@30': r1[2],
        'persistent@40': r1[3],
        'emerging@10': r2[0],
        'emerging@20': r2[1],
        'emerging@30': r2[2],
        'emerging@40': r2[3],
    }
    df = pd.DataFrame(result, index=[0])
    return df


def eval_hf(model, dataset, dataset_name, task_name, train_index, historical):
    model.eval()
    labels = dataset.label()
    outputs = []
    preds = []
    for step in range(len(dataset)):
        code_x, visit_lens, divided, y, neighbors = dataset[step]
        output = model(code_x, divided, neighbors, visit_lens).squeeze()
        output = output.detach().cpu().numpy()
        outputs.append(output)
        pred = (output > 0.5).astype(int)
        preds.append(pred)
        print('\r    Evaluating step %d / %d' % (step + 1, len(dataset)), end='')
    # pred = (output > 0.5).astype(int)
    # auc = roc_auc_score(labels, output)
    # f1_score_ = f1_score(labels, pred)
    outputs = np.array(outputs)
    preds = np.array(preds)
    # outputs = np.concatenate(outputs)
    # preds = np.concatenate(preds)
    auc = roc_auc_score(labels, outputs)
    f1_score_ = f1_score(labels, preds)
    print('\r    auc: %.4f --- f1_score: %.4f' % (auc, f1_score_))
    result = {
        'dataset_name' : dataset_name,
        'task_name' : task_name,
        'train_index' : train_index,
        'auc': auc,
        'f1_score': f1_score_
    }
    df = pd.DataFrame(result, index=[0])
    return df

if __name__ == '__main__':

    datasets = [ 'mimic3' , 'mimic4']  # 'mimic3' or 'eicu'
    tasks = ['h']#,'m']  # 'm' or 'h'
    use_cuda = False
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    # Model Parameters
    code_size = 48
    graph_size = 32
    t_attention_size = 32
    batch_size = 1
    epochs = 200

    task_conf = {
        'm': {
            'dropout': 0.45,
            'evaluate_fn': eval_diag,
            'hidden_size_mimic3': 256,
            'hidden_size_mimic4': 350,
        },
        'h': {
            'dropout': 0.0,
            'evaluate_fn': eval_hf,
            'hidden_size_mimic3': 100,
            'hidden_size_mimic4': 150,
        }
    }
    restask_h =[]
    restask_m = []
    for dataset in datasets:
        for task in tasks:
            for idx, seed in enumerate(seeds):
                print('loading test data ...')
                dataset_path = os.path.join('..','data', dataset, 'standard', str(idx))
                test_path = os.path.join(dataset_path, 'test')
                train_path = os.path.join(dataset_path, 'train')
                valid_path = os.path.join(dataset_path, 'valid')
                print('loading code_adj ...')
                code_adj = load_adj(dataset_path, device=device)
                code_num = len(code_adj)
                print(f'code_adj size {code_num}')

                test_data = EHRDataset(test_path, label=task, batch_size=batch_size, shuffle=False, device=device)
                # test_historical = historical_hot(test_data.code_x, code_num, test_data.visit_lens)
                train_data = EHRDataset(train_path, label=task, batch_size=batch_size, shuffle=False, device=device)
                valid_data = EHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)

                print(f'train {np.count_nonzero(train_data.y == 1)}')
                print(f'test {np.count_nonzero(test_data.y == 1)}')
                print(f'valid {np.count_nonzero(valid_data.y == 1)}')

                if dataset == 'mimic3':
                    hidden_size = task_conf[task]['hidden_size_mimic3']
                else:
                    hidden_size = task_conf[task]['hidden_size_mimic4']

                activation = torch.nn.Sigmoid()
                evaluate_fn = task_conf[task]['evaluate_fn']
                dropout_rate = task_conf[task]['dropout']

                if task == 'm':
                    output_size = code_num
                else:
                    output_size = 1

                t_output_size = hidden_size
                model = Model(code_num=code_num, code_size=code_size,
                              adj=code_adj, graph_size=graph_size, hidden_size=hidden_size, t_attention_size=t_attention_size,
                              t_output_size=t_output_size,
                              output_size=output_size, dropout_rate=dropout_rate, activation=activation).to(device)

                param_path = os.path.join('..','data', 'params', dataset, task,'0')

                indices = [30]#,50,75,100]

                for index in indices:
                    model_to_load = os.path.join(param_path, '%d.pt' % (index-1) )
                    model.load_state_dict(torch.load(model_to_load, map_location=device))
                    res = evaluate_fn(model, test_data, dataset, task, index, test_historical )
                    if task == 'h':
                        restask_h.append(res)

                    else:
                        restask_m.append(res)

    output_dir = os.path.join('..','out')
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    df_task_h = pd.concat(restask_h)
    df_task_m = pd.concat(restask_m)

    result_task_h = os.path.join(output_dir, 'result_task_h.csv')
    result_task_m = os.path.join(output_dir, 'result_task_m.csv')
    df_task_h.to_csv(result_task_h, index=False)
    df_task_m.to_csv(result_task_m, index=False)

