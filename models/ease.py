import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import EaseNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

num_workers = 8

import wandb

class Learner(BaseLearner):

    # ── Memory helpers ─────────────────────────────────────────────────────────
    def _cuda_sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(self._device)

    def _reset_peaks(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self._device)

    @staticmethod
    def _bytes_of_tensor_list(tensors):
        tot = 0
        for t in tensors:
            if torch.is_tensor(t) and t.is_cuda:
                tot += t.numel() * t.element_size()
        return tot

    def _bytes_model_weights(self, model: nn.Module) -> int:
        ps = [p.data for p in model.parameters() if p.is_cuda]
        bs = [b.data for b in model.buffers() if b.is_cuda]
        return self._bytes_of_tensor_list(ps + bs)

    def _bytes_grads(self, model: nn.Module) -> int:
        gs = []
        for p in model.parameters():
            g = p.grad
            if g is not None and g.is_cuda:
                gs.append(g)
        return self._bytes_of_tensor_list(gs)

    def _bytes_optimizer_states(self, optimizer: optim.Optimizer) -> int:
        tot = 0
        for st in optimizer.state.values():
            for v in st.values():
                if torch.is_tensor(v) and v.is_cuda:
                    tot += v.numel() * v.element_size()
        return tot

    @staticmethod
    def _gb(x_bytes: int) -> float:
        return x_bytes / (1024 ** 3)

    @staticmethod
    def _mb(x_bytes: int) -> float:
        return x_bytes / (1024 ** 2)

    def _log_mem_buckets(self, *, tag: str,
                         weights_b: int, grads_b: int, opt_b: int,
                         fwd_peak_b: int, peak_b: int):
        """
        Buckets measured around the *first batch of the epoch*:
          - weights_b: model/buffer tensors on CUDA
          - grads_b:   .grad tensors
          - opt_b:     optimizer state tensors (e.g., momentum, Adam m/v)
          - optimizer = grads_b + opt_b
          - activations: max(forward-only caches, residual at global peak)
        """
        # bytes → GB
        weights_gb = self._gb(weights_b)
        grads_gb = self._gb(grads_b)
        opt_gb = self._gb(opt_b)
        optim_gb = self._gb(grads_b + opt_b)

        # forward-only & residual activations
        activ_fwd_b = max(0, fwd_peak_b - weights_b)
        activ_resid_b = max(0, peak_b - (weights_b + grads_b + opt_b))
        activ_b = max(activ_fwd_b, activ_resid_b)

        activ_fwd_gb = self._gb(activ_fwd_b)
        activ_resid_gb = self._gb(activ_resid_b)
        activ_gb = self._gb(activ_b)
        fwd_peak_gb = self._gb(fwd_peak_b)
        peak_gb = self._gb(peak_b)

        # reserved (nsmi-ish) + allocator overhead
        if torch.cuda.is_available():
            res_peak_b = torch.cuda.max_memory_reserved(self._device)
        else:
            res_peak_b = 0
        overhead_b = max(0, res_peak_b - peak_b)

        # Log to W&B (GB + MiB for small values)
        wandb.log({
            f"{tag}/weights_gb": weights_gb,
            f"{tag}/grads_gb": grads_gb,
            f"{tag}/opt_states_gb": opt_gb,
            f"{tag}/optimizer_gb": optim_gb,
            f"{tag}/grads_mib": self._mb(grads_b),
            f"{tag}/opt_states_mib": self._mb(opt_b),
            f"{tag}/optimizer_mib": self._mb(grads_b + opt_b),
            f"{tag}/activations_gb": activ_gb,
            f"{tag}/activ_fwd_only_gb": activ_fwd_gb,
            f"{tag}/activ_resid_gb": activ_resid_gb,
            f"{tag}/fwd_peak_total_gb": fwd_peak_gb,
            f"{tag}/peak_total_gb": peak_gb,
            f"{tag}/peak_total_allocated_gb": self._gb(peak_b),
            f"{tag}/peak_total_reserved_gb": self._gb(res_peak_b),
            f"{tag}/overhead_non_tensor_gb": self._gb(overhead_b),
        }, step=self.global_step)

        # And print to console
        logging.info(
            "[%s] weights=%.2f GB | grads=%.6f GB (%.2f MiB) | opt_states=%.6f GB (%.2f MiB) | "
            "optimizer=%.6f GB (%.2f MiB) | activ_fwd_only=%.2f GB | activ_resid=%.2f GB | "
            "activations=%.2f GB | fwd_peak_total=%.2f GB | peak_total=%.2f GB | "
            "peak_allocated=%.2f GB | peak_reserved=%.2f GB | overhead=%.2f GB",
            tag,
            weights_gb,
            grads_gb, self._mb(grads_b),
            opt_gb, self._mb(opt_b),
            optim_gb, self._mb(grads_b + opt_b),
            activ_fwd_gb, activ_resid_gb, activ_gb,
            fwd_peak_gb, peak_gb,
            self._gb(peak_b), self._gb(res_peak_b), self._gb(overhead_b),
        )

    # ───────────────────────────────────────────────────────────────────────────

    def __init__(self, args):
        super().__init__(args)
        self._network = EaseNet(args, True)
        
        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.use_exemplars = args["use_old_data"]
        self.use_init_ptm = args["use_init_ptm"]
        self.use_diagonal = args["use_diagonal"]
        
        self.recalc_sim = args["recalc_sim"]
        self.alpha = args["alpha"] # forward_reweight is divide by _cur_task
        self.beta = args["beta"]

        self.moni_adam = args["moni_adam"]
        self.adapter_num = args["adapter_num"]
        
        if self.moni_adam:
            self.use_init_ptm = True
            self.alpha = 1 
            self.beta = 1

        # ---------- Zeroth-order hyperparams (SPSA) ----------
        # If using "zo_sgd" or "zo_adam", we’ll read them from args
        self.eps = args.get("eps", 0.001)     # small finite-difference step
        self.q_spsa = args.get("q_spsa", 1)   # how many random directions each mini-batch
        # -----------------------------------------------------

        wandb.init(project="ease_mem", config=args)
        self.global_step = 0


    def after_task(self):
        self._known_classes = self._total_classes
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()
    
    def get_cls_range(self, task_id):
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc
        
        return start_cls, end_cls
        
    # (proxy_fc = cls * dim)
    def replace_fc(self, train_loader):
        model = self._network
        model = model.eval()
        
        with torch.no_grad():           
            # replace proto for each adapter in the current task
            if self.use_init_ptm:
                start_idx = -1
            else:
                start_idx = 0
            
            for index in range(start_idx, self._cur_task + 1):
                if self.moni_adam:
                    if index > self.adapter_num - 1:
                        break
                # only use the diagonal feature, index = -1 denotes using init PTM, index = self._cur_task denotes the last adapter's feature
                elif self.use_diagonal and index != -1 and index != self._cur_task:
                    continue

                embedding_list, label_list = [], []
                for i, batch in enumerate(train_loader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=index)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())

                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                
                class_list = np.unique(self.train_dataset_for_protonet.labels)
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    if self.use_init_ptm:
                        model.fc.weight.data[class_index, (index+1)*self._network.out_dim:(index+2)*self._network.out_dim] = proto
                    else:
                        model.fc.weight.data[class_index, index*self._network.out_dim:(index+1)*self._network.out_dim] = proto

            if self.use_exemplars and self._cur_task > 0:
                embedding_list = []
                label_list = []
                dataset = self.data_manager.get_dataset(np.arange(0, self._known_classes), source="train", mode="test", )
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
                for i, batch in enumerate(loader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=self._cur_task)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())
                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                
                class_list = np.unique(dataset.labels)
                for class_index in class_list:
                    # print('adapter index:{}, Replacing...{}'.format(self._cur_task, class_index))
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    model.fc.weight.data[class_index, -self._network.out_dim:] = proto                
        
        if self.use_diagonal or self.use_exemplars:
            return
        
        if self.recalc_sim:
            self.solve_sim_reset()
        else:
            self.solve_similarity()
    
    def get_A_B_Ahat(self, task_id):
        if self.use_init_ptm:
            start_dim = (task_id + 1) * self._network.out_dim
            end_dim = start_dim + self._network.out_dim
        else:
            start_dim = task_id * self._network.out_dim
            end_dim = start_dim + self._network.out_dim
        
        start_cls, end_cls = self.get_cls_range(task_id)
        
        # W(Ti)  i is the i-th task index, T is the cur task index, W is a T*T matrix
        A = self._network.fc.weight.data[self._known_classes:, start_dim : end_dim]
        # W(TT)
        B = self._network.fc.weight.data[self._known_classes:, -self._network.out_dim:]
        # W(ii)
        A_hat = self._network.fc.weight.data[start_cls : end_cls, start_dim : end_dim]
        
        return A.cpu(), B.cpu(), A_hat.cpu()
    
    def solve_similarity(self):       
        for task_id in range(self._cur_task):          
            # print('Solve_similarity adapter:{}'.format(task_id))
            start_cls, end_cls = self.get_cls_range(task_id=task_id)

            A, B, A_hat = self.get_A_B_Ahat(task_id=task_id)
            
            # calculate similarity matrix between A_hat(old_cls1) and A(new_cls1).
            similarity = torch.zeros(len(A_hat), len(A))
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)
            
            # softmax the similarity, it will be failed if not use it
            similarity = F.softmax(similarity, dim=1)
                        
            # weight the combination of B(new_cls2)
            B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    B_hat[i] += similarity[i][j] * B[j]
            
            # B_hat(old_cls2)
            self._network.fc.weight.data[start_cls : end_cls, -self._network.out_dim:] = B_hat.to(self._device)
    
    def solve_sim_reset(self):
        for task_id in range(self._cur_task):
            if self.moni_adam and task_id > self.adapter_num - 2:
                break
            
            if self.use_init_ptm:
                range_dim = range(task_id + 2, self._cur_task + 2)
            else:
                range_dim = range(task_id + 1, self._cur_task + 1)
            for dim_id in range_dim:
                if self.moni_adam and dim_id > self.adapter_num:
                    break
                # print('Solve_similarity adapter:{}, {}'.format(task_id, dim_id))
                start_cls, end_cls = self.get_cls_range(task_id=task_id)

                start_dim = dim_id * self._network.out_dim
                end_dim = (dim_id + 1) * self._network.out_dim
                
                # Use the element above the diagonal to calculate
                if self.use_init_ptm:
                    start_cls_old = self.init_cls + (dim_id - 2) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = (task_id + 1) * self._network.out_dim
                    end_dim_old = (task_id + 2) * self._network.out_dim
                else:
                    start_cls_old = self.init_cls + (dim_id - 1) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = task_id * self._network.out_dim
                    end_dim_old = (task_id + 1) * self._network.out_dim

                A = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim_old:end_dim_old].cpu()
                B = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim:end_dim].cpu()
                A_hat = self._network.fc.weight.data[start_cls:end_cls, start_dim_old:end_dim_old].cpu()
                
                # calculate similarity matrix between A_hat(old_cls1) and A(new_cls1).
                similarity = torch.zeros(len(A_hat), len(A))
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)
                
                # softmax the similarity, it will be failed if not use it
                similarity = F.softmax(similarity, dim=1) # dim=1, not dim=0
                            
                # weight the combination of B(new_cls2)
                B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        B_hat[i] += similarity[i][j] * B[j]
                
                # B_hat(old_cls2)
                self._network.fc.weight.data[start_cls : end_cls, start_dim : end_dim] = B_hat.to(self._device)
        
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        # self._network.show_trainable_params()
        
        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train", )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        
        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        
        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self.replace_fc(self.train_loader_for_protonet)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        
        # Decide how many epochs
        if self._cur_task == 0 or self.init_cls == self.inc:
            epochs = self.args['init_epochs']
            lr = self.args['init_lr']
        else:
            epochs = self.args.get('later_epochs', self.args['init_epochs'])
            lr = self.args.get('later_lr', self.args['init_lr'])

        # 1) Construct an optimizer
        optimizer = self.get_optimizer(lr)
        scheduler = self.get_scheduler(optimizer, epochs)

        # 2) Branch to FO or ZO
        if self.args['optimizer'] in ['zo_sgd', 'zo_adam']:
            ### NEW (ZO) ###
            logging.info("Using Zeroth-Order SPSA training.")
            self._zeroth_order_train(train_loader, test_loader, optimizer, scheduler, epochs)
        else:
            logging.info("Using standard first-order training.")
            self._init_train(train_loader, test_loader, optimizer, scheduler)
    
    def get_optimizer(self, lr):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                momentum=0.9, 
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr, 
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr, 
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'zo_sgd':
            # same as SGD, but we won't actually use .backward() in ZO
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'zo_adam':
            # same as Adam, but we use SPSA to fill .grad
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer {self.args['optimizer']}")
        return optimizer
    
    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if self.moni_adam:
            if self._cur_task > self.adapter_num - 1:
                return

        if self._cur_task == 0 or self.init_cls == self.inc:
            epochs = self.args['init_epochs']
        else:
            epochs = self.args['later_epochs']

        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            # reset peak memory usage
            #torch.cuda.reset_peak_memory_stats(self._device)

            # reset peaks & set clean baseline for this epoch
            self._reset_peaks()
            optimizer.zero_grad(set_to_none=True)
            self._cuda_sync()

            # bytes of model weights (no grads/states yet for the first step)
            weights_b = self._bytes_model_weights(self._network)
            measured_this_epoch = False

            self._network.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,
                    aux_targets - self._known_classes,
                    -1,
                )

                output = self._network(inputs, test=False)
                logits = output["logits"]

                loss = F.cross_entropy(logits, aux_targets)

                # forward-only peak on the first batch
                if not measured_this_epoch and i == 0:
                    self._cuda_sync()
                    fwd_peak_b = torch.cuda.max_memory_allocated(self._device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # global peak (after backward/step) + buckets
                if not measured_this_epoch and i == 0:
                    self._cuda_sync()
                    peak_b = torch.cuda.max_memory_allocated(self._device)
                    grads_b = self._bytes_grads(self._network)
                    opt_b = self._bytes_optimizer_states(optimizer)
                    self._log_mem_buckets(
                        tag=f"mem/FO_epoch{epoch + 1}",
                        weights_b=weights_b, grads_b=grads_b, opt_b=opt_b,
                        fwd_peak_b=fwd_peak_b, peak_b=peak_b
                    )
                    measured_this_epoch = True

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

            if scheduler:
                scheduler.step()

            train_loss = losses / len(train_loader)
            train_acc = (100.0 * correct / total) if total > 0 else 0.0

            peak_mem = torch.cuda.max_memory_allocated(self._device) / (1024 ** 2)

            # log to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss_fo": train_loss,
                "train_acc_fo": train_acc,
                "peak_mem_MB_fo": peak_mem
            }, step=self.global_step)

            info = (f"[FO] Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Train_Loss={train_loss:.3f}, Train_Acc={train_acc:.2f}, "
                    f"PeakMem={peak_mem:.2f}MB")
            prog_bar.set_description(info)
            logging.info(info)

        logging.info(f"[FO] Done training for {epochs} epochs.")


    ### NEW (ZO) ###
    def _zeroth_order_train(self, train_loader, test_loader, optimizer, scheduler, epochs):
        """
        ZO training with exactly the same partial-label approach as FO (no ignore_index).
        Also log peak memory usage each epoch.
        """
        prog_bar = tqdm(range(epochs))
        for epoch_i in prog_bar:
            ########################################################################
            # Reset peak memory stats at the start of each epoch
            torch.cuda.reset_peak_memory_stats(self._device)
            ########################################################################


            self._network.train()
            total_loss = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                # partial labeling => old classes -> -1, new classes -> 0..?
                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,
                    aux_targets - self._known_classes,
                    -1,
                )

                # Collect trainable parameters
                param_list = [p for p in self._network.parameters() if p.requires_grad]
                param_grads = [torch.zeros_like(p.data) for p in param_list]

                # Q-SPSA
                for _ in range(self.q_spsa):
                    deltas = []
                    for p in param_list:
                        delta = torch.randint(0, 2, p.shape, device=self._device, dtype=p.dtype)
                        delta = delta * 2 - 1
                        deltas.append(delta)

                    # Evaluate loss_plus at (theta + eps*deltas)
                    for p, d in zip(param_list, deltas):
                        p.data.add_(self.eps * d)

                    with torch.no_grad():
                        logits_plus = self._network(inputs)["logits"]
                        loss_plus = F.cross_entropy(logits_plus, aux_targets)

                    # Evaluate loss_minus at (theta - eps*deltas)
                    for p, d in zip(param_list, deltas):
                        p.data.sub_(2.0 * self.eps * d)

                    with torch.no_grad():
                        logits_minus = self._network(inputs)["logits"]
                        loss_minus = F.cross_entropy(logits_minus, aux_targets)

                    # restore original
                    for p, d in zip(param_list, deltas):
                        p.data.add_(self.eps * d)

                    diff = (loss_plus - loss_minus)
                    for idx, d in enumerate(deltas):
                        param_grads[idx].add_(diff * d)

                # scale
                scale_factor = 1.0 / (2.0 * self.eps * self.q_spsa)
                for idx, p in enumerate(param_list):
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    p.grad.detach_()
                    p.grad.copy_(param_grads[idx] * scale_factor)


                torch.nn.utils.clip_grad_norm_(param_list, max_norm=self.args.get("max_norm", 1.0))

                optimizer.step()
                optimizer.zero_grad()



                # Evaluate batch loss & acc with same approach as FO
                #with torch.no_grad():
                #    logits_current = self._network(inputs)["logits"]
                #    loss_batch = F.cross_entropy(logits_current, aux_targets)
                #    total_loss += loss_batch.item()

                #    _, preds = torch.max(logits_current, dim=1)
                #    correct += preds.eq(aux_targets).sum().item()
                #    total += len(aux_targets)

                #self.global_step += 1
                #wandb.log({"zo_batch_loss": loss_batch.item()}, step=self.global_step)

            # End of epoch
            if scheduler:
                scheduler.step()

            ########################################################################
            # Query peak GPU usage in MB after epoch finishes
            peak_mem = torch.cuda.max_memory_allocated(self._device) / (1024 ** 2)
            ########################################################################

            train_loss = total_loss / len(train_loader)
            train_acc = (100.0 * correct / total) if total > 0 else 0.0


            wandb.log({
                "epoch": epoch_i + 1,
                "train_loss_zo": train_loss,
                "train_acc_zo": train_acc,
                "peak_mem_MB_zo": peak_mem
            }, step=self.global_step)

            info = (f"[ZO] Task {self._cur_task}, Epoch {epoch_i + 1}/{epochs} => "
                    f"Train_Loss={train_loss:.3f}, Train_Acc={train_acc:.2f}, "
                    f"PeakMem={peak_mem:.2f}MB")
            prog_bar.set_description(info)
            logging.info(info)

        logging.info(f"[ZO] Finished SPSA training after {epochs} epochs.")



    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]          
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        calc_task_acc = True
        
        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0
            
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            
            # calculate the accuracy by using task_id
            if calc_task_acc:
                task_ids = (targets - self.init_cls) // self.inc + 1
                task_logits = torch.zeros(outputs.shape).to(self._device)
                for i, task_id in enumerate(task_ids):
                    if task_id == 0:
                        start_cls = 0
                        end_cls = self.init_cls
                    else:
                        start_cls = self.init_cls + (task_id-1)*self.inc
                        end_cls = self.init_cls + task_id*self.inc
                    task_logits[i, start_cls:end_cls] += outputs[i, start_cls:end_cls]
                # calculate the accuracy of task_id
                pred_task_ids = (torch.max(outputs, dim=1)[1] - self.init_cls) // self.inc + 1
                task_correct += (pred_task_ids.cpu() == task_ids).sum()
                
                pred_task_y = torch.max(task_logits, dim=1)[1]
                task_acc += (pred_task_y.cpu() == targets).sum()
                total += len(targets)

        if calc_task_acc:
            logging.info("Task correct: {}".format(tensor2numpy(task_correct) * 100 / total))
            logging.info("Task acc: {}".format(tensor2numpy(task_acc) * 100 / total))
                
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]