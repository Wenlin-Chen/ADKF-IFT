import math
import random
import os
from typing import List
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

# Custom FS mol
import sys
sys.path.append("../")
from bayes_opt import bo_utils
from bayes_opt.bo_utils import task_to_batches
from rdkit.Chem import CanonSmiles, MolFromSmiles, rdFingerprintGenerator, Descriptors
from rdkit import DataStructs
from fs_mol.preprocessing.featurisers.molgraph_utils import molecule_to_graph
from fs_mol.data.fsmol_task import MoleculeDatapoint, GraphData
from fs_mol.data import (
    FSMolTask,
    FSMolBatch,
    FSMolBatcher,
    MoleculeDatapoint,
)
from fs_mol.data.dkt import get_dkt_batcher
from fs_mol.utils.torch_utils import torchify

from botorch.optim.fit import fit_gpytorch_scipy

class Meta_Trainer(nn.Module):
    def __init__(self, args, model):
        super(Meta_Trainer, self).__init__()

        self.args = args

        self.model = model
        self.optimizer = None #optim.AdamW(self.model.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
        # self.criterion = nn.CrossEntropyLoss().to(args.device)

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

        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.inner_update_step = args.inner_update_step

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

    def get_prediction(self, model, data, train=True):
        raise NotImplementedError
        if train:
            s_logits, q_logits, adj, node_emb = model(data['s_data'], data['q_data'], data['s_label'])
            pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'adj': adj, 'node_emb': node_emb}

        else:
            s_logits, logits,labels,adj_list,sup_labels = model.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'])
            pred_dict = {'s_logits':s_logits, 'logits': logits, 'labels': labels,'adj':adj_list,'sup_labels':sup_labels}

        return pred_dict

    # def get_adaptable_weights(self, model, adapt_weight=None):
    #     if adapt_weight is None:
    #         adapt_weight = self.args.adapt_weight
    #     fenc = lambda x: x[0]== 'mol_encoder'
    #     frel = lambda x: x[0]== 'adapt_relation'
    #     fedge = lambda x: x[0]== 'adapt_relation' and 'edge_layer'  in x[1]
    #     fnode = lambda x: x[0]== 'adapt_relation' and 'node_layer'  in x[1]
    #     fclf = lambda x: x[0]== 'adapt_relation' and 'fc'  in x[1]
    #     if adapt_weight==0:
    #         flag=lambda x: not fenc(x)
    #     elif adapt_weight==1:
    #         flag=lambda x: not frel(x)
    #     elif adapt_weight==2:
    #         flag=lambda x: not (fenc(x) or frel(x))
    #     elif adapt_weight==3:
    #         flag=lambda x: not (fenc(x) or fedge(x))
    #     elif adapt_weight==4:
    #         flag=lambda x: not (fenc(x) or fnode(x))
    #     elif adapt_weight==5:
    #         flag=lambda x: not (fenc(x) or fnode(x) or fedge(x))
    #     elif adapt_weight==6:
    #         flag=lambda x: not (fenc(x) or fclf(x))
    #     else:
    #         flag= lambda x: True
    #     if self.train_epoch < self.args.meta_warm_step or self.train_epoch>self.args.meta_warm_step2:
    #         adaptable_weights = None
    #     else:
    #         adaptable_weights = []
    #         adaptable_names=[]
    #         for name, p in model.module.named_parameters():
    #             names=name.split('.')
    #             if flag(names):
    #                 adaptable_weights.append(p)
    #                 adaptable_names.append(name)
    #     return adaptable_weights

    def get_loss(self, model, batch_data, pred_dict, train=True, flag = 0):
        raise NotImplementedError

        n_support_train = self.args.n_shot_train
        n_support_test = self.args.n_shot_test
        n_query = self.args.n_query
        if not train:
            losses_adapt = self.criterion(pred_dict['s_logits'].reshape(2*n_support_test*n_query,2), batch_data['s_label'].repeat(n_query))
        else:
            if flag:
                losses_adapt = self.criterion(pred_dict['s_logits'].reshape(2*n_support_train*n_query,2), batch_data['s_label'].repeat(n_query))
            else:
                losses_adapt = self.criterion(pred_dict['q_logits'], batch_data['q_label'])

        if torch.isnan(losses_adapt).any() or torch.isinf(losses_adapt).any():
            print('!!!!!!!!!!!!!!!!!!! Nan value for supervised CE loss', losses_adapt)
            print(pred_dict['s_logits'])
            losses_adapt = torch.zeros_like(losses_adapt)
        if self.args.reg_adj > 0:
            n_support = batch_data['s_label'].size(0)
            adj = pred_dict['adj'][-1]
            if train:
                if flag:
                    s_label = batch_data['s_label'].unsqueeze(0).repeat(n_query, 1)
                    n_d = n_query * n_support
                    label_edge = model.label2edge(s_label).reshape((n_d, -1))
                    pred_edge = adj[:,:,:-1,:-1].reshape((n_d, -1))
                else:
                    s_label = batch_data['s_label'].unsqueeze(0).repeat(n_query, 1)
                    q_label = batch_data['q_label'].unsqueeze(1)
                    total_label = torch.cat((s_label, q_label), 1)
                    label_edge = model.label2edge(total_label)[:,:,-1,:-1]
                    pred_edge = adj[:,:,-1,:-1]
            else:
                s_label = batch_data['s_label'].unsqueeze(0)
                n_d = n_support
                label_edge = model.label2edge(s_label).reshape((n_d, -1))
                pred_edge = adj[:, :, :n_support, :n_support].mean(0).reshape((n_d, -1))
            adj_loss_val = F.mse_loss(pred_edge, label_edge)
            if torch.isnan(adj_loss_val).any() or torch.isinf(adj_loss_val).any():
                print('!!!!!!!!!!!!!!!!!!!  Nan value for adjacency loss', adj_loss_val)
                adj_loss_val = torch.zeros_like(adj_loss_val)

            losses_adapt += self.args.reg_adj * adj_loss_val

        return losses_adapt

    def train_step(self):
        raise NotImplementedError


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
            losses_eval = []
            for task_id in task_id_list:
                train_data, _ = data_batches[task_id]
                model = self.model.clone()
                model.train()
                adaptable_weights = self.get_adaptable_weights(model)
                
                for inner_step in range(self.inner_update_step):
                    pred_adapt = self.get_prediction(model, train_data, train=True)
                    loss_adapt = self.get_loss(model, train_data, pred_adapt, train=True, flag = 1)
                    model.adapt(loss_adapt, adaptable_weights = adaptable_weights)

                pred_eval = self.get_prediction(model, train_data, train=True)
                loss_eval = self.get_loss(model, train_data, pred_eval, train=True, flag = 0)

                losses_eval.append(loss_eval)

            losses_eval = torch.stack(losses_eval)

            losses_eval = torch.sum(losses_eval)

            losses_eval = losses_eval / len(task_id_list)
            self.optimizer.zero_grad()
            losses_eval.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            print('Train Epoch:',self.train_epoch,', train update step:', k, ', loss_eval:', losses_eval.item())

        return self.model.module


    def get_task(self, labels, smiles_list, task_name):
        dataset = []

        # get pre-defined atom_feature_extractors from metadata provided in FS-Mol
        metadata_path =  "../fs_mol/preprocessing/utils/helper_files/"
        atom_feature_extractors = bo_utils.get_feature_extractors_from_metadata(metadata_path)

        for i, (smiles, label) in enumerate(zip(smiles_list, labels)):

            # get molecule info from the xlsx file
            if smiles == "FAIL":
                print(f'Warning: smiles {smiles} excluded')
                continue
            smiles = CanonSmiles(smiles)
            bool_label = True if math.isclose(label, 1.0) else False
            
            # get fingerprint
            rdkit_mol = MolFromSmiles(smiles)
            fp_vec = rdFingerprintGenerator.GetCountFPs([rdkit_mol], fpType=rdFingerprintGenerator.MorganFP)[0]
            fp_numpy = np.zeros((0,), np.int8) 
            DataStructs.ConvertToNumpyArray(fp_vec, fp_numpy)
            
            # get graph
            try:
                graph_dict = molecule_to_graph(rdkit_mol, atom_feature_extractors)
                adjacency_lists = []
                for adj_list in graph_dict["adjacency_lists"]:
                    if adj_list:
                        adjacency_lists.append(np.array(adj_list, dtype=np.int64))
                    else:
                        adjacency_lists.append(np.zeros(shape=(0, 2), dtype=np.int64))
                graph = GraphData(
                    node_features=np.array(graph_dict["node_features"], dtype=np.float32), 
                    adjacency_lists=adjacency_lists, 
                    edge_features=[]
                )
            except IndexError:
                #print(
                #    f"Skipping datapoint {smiles}, cannot featurise with current metadata."
                #)
                print(f'Warning: smiles {smiles} excluded')
                # raise ValueError
                continue

            # get descriptors
            descriptors = []
            for descr in Descriptors._descList:
                _, descr_calc_fn = descr
                descriptors.append(descr_calc_fn(rdkit_mol))
            descriptors = np.array(descriptors, dtype=np.float32)
            
            # create a MoleculeDatapoint object
            mol = MoleculeDatapoint(
                task_name="antibiotics", smiles=smiles, graph=graph, 
                numeric_label=math.nan, bool_label=bool_label, 
                fingerprint=fp_numpy, descriptors=descriptors)
            dataset.append(mol)

        return FSMolTask(name=task_name, samples=dataset)

    def test_step(self):
        step_results={'query_preds':[], 'query_labels':[], 'query_adj':[],'task_index':[]}
        auc_scores = []
        for task_id in range(len(self.test_tasks)):
            adapt_data, eval_data = self.get_data_sample(task_id, train=False)
            # model = self.model.clone()

            # Get representations for support set
            task_adapt = self.get_task(
                labels=list(adapt_data['s_data'].y.cpu().numpy()),
                smiles_list=adapt_data['s_data'].smiles,
                task_name=str(task_id)
            )

            # Get batches for support set
            batcher_adapt = get_dkt_batcher(max_num_graphs=100)
            batches_adapt = torchify(
                task_to_batches(task_adapt, batcher_adapt), 
                device=self.args.device
            )

            # Get representations
            self.model.eval()
            representations_adapt = [self.model.get_representation(batch) for batch in batches_adapt]

            # Make training data
            x_train = torch.cat(representations_adapt, dim=0)
            y_train = torch.FloatTensor([float(x.bool_label) for x in task_adapt.samples]).to(self.args.device)

            # Fit GP
            y_train = y_train * 2 - 1

            likelihood, model, mll = bo_utils.create_gp(x_train, y_train, "matern", self.args.device, 0.01, True)
            model.train()
            likelihood.train()
            fit_gpytorch_scipy(mll)
            model.eval()
            likelihood.eval()

            y_test_preds = []
            y_test_trues = []
            # For query set
            for query_batch in list(eval_data['data_loader']):

                task_eval = self.get_task(
                    labels=list(query_batch.y.cpu().numpy()),
                    smiles_list=query_batch.smiles,
                    task_name=str(task_id)
                )

                # Get batches for query set
                batcher_eval = get_dkt_batcher(max_num_graphs=100)
                batches_eval = torchify(
                    task_to_batches(task_eval, batcher_eval),
                    device=self.args.device
                )

                # Get representations
                self.model.eval()
                representations_eval = [self.model.get_representation(batch) for batch in batches_eval]

                # Make test data
                x_test = torch.cat(representations_eval, dim=0)

                with torch.no_grad():
                    y_test_pred = likelihood(model(x_test)).loc
                #y_test_pred = (y_test_pred + 1) / 2
                y_test_pred = torch.sigmoid(y_test_pred)

                y_test_preds.append(y_test_pred)
                y_test_trues.extend([x.bool_label for x in task_eval.samples])

            y_test_preds = torch.concat(y_test_preds)
            y_test_trues = torch.as_tensor(y_test_trues, dtype=torch.long).to(y_test_preds.device)

            # Get AUC scores
            with torch.no_grad():
                if self.args.eval_support:
                    raise NotImplementedError
                    y_s_score = F.softmax(pred_eval['s_logits'],dim=-1).detach()[:,1]
                    y_s_true = eval_data['s_label']
                    y_score=torch.cat([y_score, y_s_score])
                    y_true=torch.cat([y_true, y_s_true])
                auc = auroc(y_test_preds,y_test_trues,pos_label=1).item()
                print(auc)

            auc_scores.append(auc)

            print('Test Epoch:',self.train_epoch,', test for task:', task_id, ', AUC:', round(auc, 4))
            if self.args.save_logs:
                # step_results['query_preds'].append(y_score.cpu().numpy())
                # step_results['query_labels'].append(y_true.cpu().numpy())
                step_results['query_adj'].append([])
                step_results['task_index'].append(self.test_tasks[task_id])

        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        self.best_auc = max(self.best_auc,avg_auc)
        self.logger.append([self.train_epoch] + auc_scores  +[avg_auc, mid_auc,self.best_auc], verbose=False)

        print('Test Epoch:', self.train_epoch, ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
            ', Best_Avg_AUC: ', round(self.best_auc, 4),)
        
        if self.args.save_logs:
            self.res_logs.append(step_results)

        return self.best_auc

    def save_model(self):
        save_path = os.path.join(self.trial_path, f"step_{self.train_epoch}.pth")
        torch.save(self.model.module.state_dict(), save_path)
        print(f"Checkpoint saved in {save_path}")

    def save_result_log(self):
        joblib.dump(self.res_logs,self.args.trial_path+'/logs.pkl',compress=6)

    def conclude(self):
        df = self.logger.conclude()
        self.logger.close()
        print(df)
