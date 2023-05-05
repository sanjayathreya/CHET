import os
import random
import time

import torch
import numpy as np
import pandas as pd
from model import Model, ModelWithoutTransition, ModelWithOnlySingleEmbedding
from utils import load_adj, EHRDataset, format_time, MultiStepLRScheduler
from metrics import evaluate_codes, evaluate_hf
from config import config


def historical_hot(code_x, code_num, lens):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        result[i] = x[l - 1]
    return result


if __name__ == '__main__':

    datasets = [ 'mimic4', 'mimic3']
    tasks = ['h', 'm']
    seeds = [6669]#, 1000, 1050]#, 2052, 3000]

    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    config, code_size, graph_size, t_attention_size, batch_size, epochs = config()

    res = []
    for dataset in datasets:
        for task in tasks:
            for idx, seed in enumerate(seeds):

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                dataset_path = os.path.join('..','data', dataset, 'standard')
                train_path = os.path.join(dataset_path, 'train')
                valid_path = os.path.join(dataset_path, 'valid')

                code_adj = load_adj(dataset_path, device=device)
                code_num = len(code_adj)
                print('loading train data ...')
                train_data = EHRDataset(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
                print('loading valid data ...')
                valid_data = EHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)


                test_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)

                if task == 'm':
                    output_size = code_num
                else:
                    output_size = 1

                activation = torch.nn.Sigmoid()
                loss_fn = torch.nn.BCELoss()
                evaluate_fn = config[task]['evaluate_fn']
                dropout_rate = config[task]['dropout']
                hidden_size = config[task]['hidden_size'][dataset]
                t_output_size = hidden_size

                param_path = os.path.join('..','data', 'params-ablation2', dataset, task, str(idx))
                if not os.path.exists(param_path):
                    os.makedirs(param_path)

                print('Calling Model without Transition')
                model = ModelWithoutTransition(code_num=code_num, code_size=code_size,
                               adj=code_adj, graph_size=graph_size, t_attention_size=t_attention_size,
                               t_output_size=t_output_size,
                               output_size=output_size, dropout_rate=dropout_rate, activation=activation).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                scheduler = MultiStepLRScheduler(optimizer, epochs, config[task]['lr']['init_lr'],
                                                 config[task]['lr']['milestones'], config[task]['lr']['lrs'])

                pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(pytorch_total_params)

                train_start_time = time.time()
                for epoch in range(epochs):
                    print('Epoch %d / %d:' % (epoch + 1, epochs))
                    model.train()
                    total_loss = 0.0
                    total_num = 0
                    steps = len(train_data)
                    st = time.time()
                    scheduler.step()
                    for step in range(len(train_data)):
                        optimizer.zero_grad()
                        code_x, visit_lens, divided, y, neighbors = train_data[step]
                        output = model(code_x, divided, neighbors, visit_lens).squeeze()
                        loss = loss_fn(output, y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * output_size * len(code_x)
                        total_num += len(code_x)

                        end_time = time.time()
                        remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))
                        print('\r    Step %d / %d, remaining time: %s, loss: %.4f'
                              % (step + 1, steps, remaining_time, total_loss / total_num), end='')
                    train_data.on_epoch_end()
                    et = time.time()
                    time_cost = format_time(et - st)
                    print('\r    Step %d / %d, time cost: %s, loss: %.4f' % (
                    steps, steps, time_cost, total_loss / total_num))
                    valid_loss, f1_score = evaluate_fn(model, valid_data, loss_fn, output_size, test_historical)
                    torch.save(model.state_dict(), os.path.join(param_path, '%d.pt' % epoch))

                train_end_time = time.time()
                result = {
                    'dataset_name': dataset,
                    'sample': task,
                    'seed#': seed,
                    'seed_idx': idx,
                    'total_params': pytorch_total_params,
                    'train_time': (train_end_time - train_start_time)
                }
                df_ = pd.DataFrame(result, index=[0])
                res.append(df_)
                print(f'\n******Completed training dataset:{dataset} task:{task} seed:{seed} idx:{idx} time: {format_time(train_end_time - train_start_time)}******\n')

    df = pd.concat(res)
    # Write preprocessing time to output directory
    output_dir = os.path.join('..','out')
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    output_file = os.path.join(output_dir,'output_training-ablation2.csv')
    df.to_csv(output_file, index=False)