from metrics import evaluate_codes, evaluate_hf

def config():
    config = {
        'm': {
            'dropout': 0.45,
            'evaluate_fn': evaluate_codes,
            'hidden_size' :{
                'mimic3': 256,
                'mimic4': 350,
            },
            'lr': {
                'init_lr': 0.01,
                'milestones': [20, 30],
                'lrs': [1e-3, 1e-5]
            }
        },
        'h': {
            'dropout': 0.0,
            'evaluate_fn': evaluate_hf,
            'hidden_size': {
                'mimic3': 100,
                'mimic4': 150,
            },
            'lr': {
                'init_lr': 0.01,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-4, 1e-5]
            }
        }
    }

    # Model Parameters
    code_size = 48
    graph_size = 32
    t_attention_size = 32
    batch_size = 32
    epochs = 31

    return config, code_size, graph_size, t_attention_size, batch_size, epochs