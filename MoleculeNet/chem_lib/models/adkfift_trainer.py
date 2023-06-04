import random
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import auroc
from torch_geometric.data import DataLoader

from .maml import MAML
from ..datasets import sample_meta_datasets, sample_test_datasets, MoleculeDataset
from ..utils import Logger

from botorch.optim.fit import fit_gpytorch_scipy

from copy import deepcopy

# Custom FS mol
import sys
sys.path.append("../")
from fs_mol.utils.cauchy_hypergradient import cauchy_hypergradient
from fs_mol.utils._stateless import functional_call

class ADKF_Meta_Trainer(nn.Module):
    def __init__(self, args, model):
        super(ADKF_Meta_Trainer, self).__init__()

        self.args = args

        #self.model = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)
        self.model = model
        self.optimizer = optim.AdamW(self.model.feature_extractor_params(), lr=args.meta_lr, weight_decay=args.weight_decay)
        #self.criterion = nn.CrossEntropyLoss().to(args.device)

        self.dataset = args.dataset
        self.test_dataset = args.test_dataset if args.test_dataset is not None else args.dataset
        self.data_dir = args.data_dir
        self.train_tasks = args.train_tasks
        self.test_tasks = args.test_tasks
        self.n_shot_train = args.n_shot_train
        self.n_shot_test = args.n_shot_test
        self.n_query = args.n_query

        self.device = args.device

        self.emb_dim = args.emb_dim

        self.batch_task = args.batch_task

        self.update_step = args.update_step  # the number of outer loops in an epoch during meta-training
        self.update_step_test = args.update_step_test  # the number of outer loops in an epoch during meta-testing
        #self.inner_update_step = args.inner_update_step

        self.trial_path = args.trial_path
        trial_name = self.dataset + '_' + self.test_dataset + '@' + args.enc_gnn
        print(trial_name)
        logger = Logger(self.trial_path + '/results.txt', title=trial_name)
        log_names = ['Epoch']
        log_names += ['AUC-' + str(t) for t in args.test_tasks]
        log_names += ['AUC-Avg', 'AUC-Mid','AUC-Best']
        logger.set_names(log_names)
        self.logger = logger

        preload_train_data = {}
        if args.preload_train_data:
            print('preload train data')
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1),
                                          dataset=self.dataset)
                preload_train_data[task] = dataset
        preload_test_data = {}
        if args.preload_test_data:
            print('preload_test_data')
            for task in self.test_tasks:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
                preload_test_data[task] = dataset
        self.preload_train_data = preload_train_data
        self.preload_test_data = preload_test_data
        if 'train' in self.dataset and args.support_valid:
            val_data_name = self.dataset.replace('train','valid')
            print('preload_valid_data')
            preload_val_data = {}
            for task in self.train_tasks:
                dataset = MoleculeDataset(self.data_dir + val_data_name + "/new/" + str(task + 1),
                                          dataset=val_data_name)
                preload_val_data[task] = dataset
            self.preload_valid_data = preload_val_data

        self.train_epoch = 0
        self.best_auc=0 
        
        self.res_logs=[]

    def loader_to_samples(self, data):
        loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)
        for samples in loader:
            samples=samples.to(self.device)
            return samples

    def get_data_sample(self, task_id, train=True):
        if train:
            task = self.train_tasks[task_id]
            if task in self.preload_train_data:
                dataset = self.preload_train_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(task + 1), dataset=self.dataset)

            s_data, q_data = sample_meta_datasets(dataset, self.dataset, task,self.n_shot_train, self.n_query)

            s_data = self.loader_to_samples(s_data)
            q_data = self.loader_to_samples(q_data)

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label': q_data.y,
                            'label': torch.cat([s_data.y, q_data.y], 0)}
            eval_data = { }
        else:
            task = self.test_tasks[task_id]
            if task in self.preload_test_data:
                dataset = self.preload_test_data[task]
            else:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(task + 1),
                                          dataset=self.test_dataset)
            s_data, q_data, q_data_adapt = sample_test_datasets(dataset, self.test_dataset, task, self.n_shot_test, self.n_query, self.update_step_test)
            s_data = self.loader_to_samples(s_data)

            q_loader = DataLoader(q_data, batch_size=self.n_query, shuffle=True, num_workers=0)
            q_loader_adapt = DataLoader(q_data_adapt, batch_size=self.n_query, shuffle=True, num_workers=0)

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader_adapt}
            eval_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader}

        return adapt_data, eval_data

    # def get_prediction(self, model, data, train=True):
    #     if train:
    #         s_logits, q_logits, adj, node_emb = model(data['s_data'], data['q_data'], data['s_label'])
    #         pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'adj': adj, 'node_emb': node_emb}

    #     else:
    #         s_logits, logits,labels, adj_list, sup_labels = model.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'])
    #         pred_dict = {'s_logits':s_logits, 'logits': logits, 'labels': labels, 'adj':adj_list, 'sup_labels':sup_labels}

    #     return pred_dict

    def train_step(self):

        self.train_epoch += 1

        task_id_list = list(range(len(self.train_tasks)))
        if self.batch_task > 0:
            batch_task = min(self.batch_task, len(task_id_list))
            task_id_list = random.sample(task_id_list, batch_task)
        data_batches={}
        for task_id in task_id_list:
            db = self.get_data_sample(task_id, train=True)
            data_batches[task_id]=db

        for k in range(self.update_step):
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()

            grad_accum = [0.0 for p in self.model.feature_extractor_params()]
            losses_eval = []

            for task_id in task_id_list:
                train_data, _ = data_batches[task_id]
                
                # inner update
                self.model.train()
                _ = self.model(
                    s_data=train_data['s_data'], q_data=None, s_label=train_data['s_label'], train_loss=True)
                fit_gpytorch_scipy(self.model.mll)

                # (outer) hypergrad computation
                self.model.train()
                feature_extractor_params_names = [n for n, _ in self.model.named_parameters() if not n.startswith("gp_")]
                #feature_extractor_params_names = [n for n, _ in self.model.named_parameters() if n.startswith("mol_encoder.gnn.gnns.4") or n.startswith("mol_encoder.gnn.batch_norms.4")]
                gp_params_names = [n for n, _ in self.model.named_parameters() if n.startswith("gp_")]
                #assert False
                def f_inner(params_outer, params_inner):
                    feature_extractor_params_dict = {n: p for n, p in zip(feature_extractor_params_names, params_outer)}
                    gp_params_dict = {n: p for n, p in zip(gp_params_names, params_inner)}
                    self_params_dict = {**feature_extractor_params_dict, **gp_params_dict}
                    batch_loss = functional_call(
                        self.model, self_params_dict, (train_data['s_data'], train_data['q_data']),
                        kwargs={"s_label": train_data['s_label'], "train_loss": True, "is_functional_call": True})
                    return batch_loss

                def f_outer(params_outer, params_inner):
                    feature_extractor_params_dict = {n: p for n, p in zip(feature_extractor_params_names, params_outer)}
                    gp_params_dict = {n: p for n, p in zip(gp_params_names, params_inner)}
                    self_params_dict = {**feature_extractor_params_dict, **gp_params_dict}
                    batch_loss = functional_call(
                        self.model, self_params_dict, (train_data['s_data'], train_data['q_data']), 
                        kwargs={"s_label": train_data['s_label'], "train_loss": False, "predictive_val_loss": True, "is_functional_call": True})
                    return batch_loss
                #assert False
                batch_loss = cauchy_hypergradient(f_outer, f_inner, tuple(self.model.feature_extractor_params()), tuple(self.model.gp_params()), self.device, ignore_grad_correction=False)
                #assert False
                losses_eval.append(batch_loss.cpu().item()/len(train_data['q_data'].x))

                for i, param in enumerate(self.model.feature_extractor_params()):
                    grad_accum[i] += param.grad.data.clone() / len(task_id_list)

            for i, param in enumerate(self.model.feature_extractor_params()):
                param.grad = grad_accum[i]

            torch.nn.utils.clip_grad_norm_(self.model.feature_extractor_params(), 1.0)
            self.optimizer.step()

            losses_eval = np.mean(losses_eval)

            print('Train Epoch:', self.train_epoch,', train update step:', k, ', loss_eval:', losses_eval)

        return self.model

    def test_step(self, args):
        saved_state_dict = deepcopy(self.model.state_dict())

        step_results={'query_preds':[], 'query_labels':[], 'task_index':[]}
        auc_scores = []
        for task_id in range(len(self.test_tasks)):
            self.model.load_state_dict(saved_state_dict)
            adapt_data, eval_data = self.get_data_sample(task_id, train=False)

            if self.update_step_test>0:
                
                for i, batch in enumerate(adapt_data['data_loader']):
                    self.optimizer.zero_grad()
                    batch = batch.to(self.device)
                    cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label'],
                                        'q_data': batch, 'q_label': None}

                    # inner update
                    self.model.train()
                    _ = self.model(
                        s_data=cur_adapt_data['s_data'], q_data=None, s_label=cur_adapt_data['s_label'], train_loss=True)
                    fit_gpytorch_scipy(self.model.mll)

                    # (outer) hypergrad computation
                    self.model.train()
                    feature_extractor_params_names = [n for n, _ in self.model.named_parameters() if not n.startswith("gp_")]
                    #feature_extractor_params_names = [n for n, _ in self.model.named_parameters() if n.startswith("mol_encoder.gnn.gnns.4") or n.startswith("mol_encoder.gnn.batch_norms.4")]
                    gp_params_names = [n for n, _ in self.model.named_parameters() if n.startswith("gp_")]
                    
                    def f_inner(params_outer, params_inner):
                        feature_extractor_params_dict = {n: p for n, p in zip(feature_extractor_params_names, params_outer)}
                        gp_params_dict = {n: p for n, p in zip(gp_params_names, params_inner)}
                        self_params_dict = {**feature_extractor_params_dict, **gp_params_dict}
                        batch_loss = functional_call(
                            self.model, self_params_dict, (cur_adapt_data['s_data'], cur_adapt_data['q_data']),
                            kwargs={"s_label": cur_adapt_data['s_label'], "train_loss": True, "is_functional_call": True})
                        return batch_loss

                    def f_outer(params_outer, params_inner):
                        feature_extractor_params_dict = {n: p for n, p in zip(feature_extractor_params_names, params_outer)}
                        gp_params_dict = {n: p for n, p in zip(gp_params_names, params_inner)}
                        self_params_dict = {**feature_extractor_params_dict, **gp_params_dict}
                        batch_loss = functional_call(
                            self.model, self_params_dict, (cur_adapt_data['s_data'], cur_adapt_data['q_data']), 
                            kwargs={"s_label": cur_adapt_data['s_label'], "train_loss": False, "predictive_val_loss": True, "is_functional_call": True})
                        return batch_loss
                    #assert False
                    batch_loss = cauchy_hypergradient(f_outer, f_inner, tuple(self.model.feature_extractor_params()), tuple(self.model.gp_params()), self.device, ignore_grad_correction=False)
                    torch.nn.utils.clip_grad_norm_(self.model.feature_extractor_params(), 1.0)
                    self.optimizer.step()
                    if i>= self.update_step_test-1:
                        break

            self.model.train()
            _ = self.model(
                s_data=eval_data['s_data'], q_data=None, s_label=eval_data['s_label'], train_loss=True)
            fit_gpytorch_scipy(self.model.mll)

            self.model.eval()
            with torch.no_grad():
                q_preds, q_labels = self.model.forward_query_loader(eval_data['s_data'], eval_data['data_loader'], train_loss=None, s_label=eval_data['s_label'])

                # if self.args.eval_support:
                #     y_s_score = F.softmax(pred_eval['s_logits'],dim=-1).detach()[:,1]
                #     y_s_true = eval_data['s_label']
                #     y_score=torch.cat([y_score, y_s_score])
                #     y_true=torch.cat([y_true, y_s_true])
                auc = auroc(q_preds, q_labels, pos_label=1).item()

            auc_scores.append(auc)

            print('Test Epoch:',self.train_epoch,', test for task:', task_id, ', AUC:', round(auc, 4))
            if self.args.save_logs:
                step_results['query_preds'].append(q_preds.cpu().numpy())
                step_results['query_labels'].append(q_labels.cpu().numpy())
                step_results['task_index'].append(self.test_tasks[task_id])

        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        self.best_auc = max(self.best_auc,avg_auc)
        self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc,self.best_auc], verbose=False)

        print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
              ', Best_Avg_AUC: ', round(self.best_auc, 4),)
        
        if self.args.save_logs:
            self.res_logs.append(step_results)

        self.model.load_state_dict(saved_state_dict)
        return self.best_auc

    def save_model(self):
        save_path = os.path.join(self.trial_path, f"step_{self.train_epoch}.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"Checkpoint saved in {save_path}")

    def save_result_log(self):
        joblib.dump(self.res_logs,self.args.trial_path+'/logs.pkl',compress=6)

    def conclude(self):
        df = self.logger.conclude()
        self.logger.close()
        print(df)
