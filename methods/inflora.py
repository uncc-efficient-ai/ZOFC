import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from models.sinet_inflora import SiNet
from models.vit_inflora import Attention_LoRA
from copy import deepcopy
from utils.schedulers import CosineSchedule
import ipdb
import math

import wandb

class InfLoRA(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        if args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))

        for module in self._network.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

        self.args = args
        self.optim = args["optim"]
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.lamb = args["lamb"]
        self.lame = args["lame"]
        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]

        self.topk = 1  # origin is 5
        self.class_num = self._network.class_num
        self.debug = False


        self.all_keys = []
        self.feature_list = []
        self.project_type = []


        # ------------------------- [MOD] ZO-SPSA parity with LAE -------------------------
        # Trainer mode: "FO" (default) or "ZO" (LAE-style SPSA); "Hybrid" not implemented here
        self.trainer = args.get("trainer", "FO").upper()
        # LAE ZO-SPSA hyperparams
        self.eps = args.get("eps", 1e-3)               # perturbation radius
        self.q_spsa = int(args.get("q_spsa", 1))       # # of SPSA directions per batch
        self.max_norm = float(args.get("max_norm", 1.0))  # clip grad norm on surrogate grads
        self.zo_loss_mode = args.get("zo_loss_mode", "current_task").lower()

        # Hybrid-specific hyperparams
        self.pet_lr = float(args.get("pet_lr", self.init_lr if self._cur_task == 0 else self.lrate))
        self.fc_lr = float(args.get("fc_lr", self.init_lr if self._cur_task == 0 else self.lrate))
        self.fc_epoch = int(args.get("fc_epoch", max(1, self.epochs // 2)))  # FO-only warmup epochs for the classifier
        self.hybrid_optimizer = args.get("optimizer", "zo_sgd").lower()  # "zo_sgd" or "zo_adam" for LoRA side
        # ----------------------------------------------------------------------------------

        # ---------------- W&B init ----------------

        wandb.init(
            project=self.args.get("wandb_project", "inflora"),
            config=self.args
        )

        # global step counter shared by both FO and ZO trainers
        self.global_step = 0

        # Make all logs advance by this step
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("zo/*", step_metric="global_step")
        wandb.define_metric("mem/*", step_metric="global_step")

        # ------------------------------------------

    def after_task(self):
        # self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)

        # if len(self._multiple_gpus) > 1:
        #     self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.clustering(self.train_loader)
        # if len(self._multiple_gpus) > 1:
        #     self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        # if self._old_network is not None:
        #     self._old_network.to(self._device)

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            try:
                if "classifier_pool" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
            except:
                if "classifier_pool" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_B_k" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                if "lora_B_v" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                self._network(inputs, get_cur_feat=True)
                # if i > 3: break

            if self._cur_task == 0:
                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix
                        U, S, V = torch.linalg.svd(cur_matrix)
                        module.lora_A_k[self._cur_task].weight.data.copy_(U[:, :module.rank].T / math.sqrt(3))
                        module.lora_A_v[self._cur_task].weight.data.copy_(U[:, :module.rank].T / math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
            else:
                # kk = 0
                # for module in self._network.modules():
                #     if isinstance(module, Attention_LoRA):
                #         cur_matrix = module.cur_matrix
                #         cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk],cur_matrix)
                #         cU, cS, cV = torch.linalg.svd(cur_matrix, full_matrices=False)
                #         module.lora_A_k[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                #         module.lora_A_v[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/math.sqrt(3))
                #         module.cur_matrix.zero_()
                #         module.n_cur_matrix = 0
                #         kk += 1

                kk = 0
                for module in self._network.modules():
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix
                        if self.project_type[kk] == 'remove':
                            cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk], cur_matrix)
                        else:
                            assert self.project_type[kk] == 'retain'
                            cur_matrix = torch.mm(self.feature_mat[kk], cur_matrix)
                        cU, cS, cV = torch.linalg.svd(cur_matrix, full_matrices=False)
                        module.lora_A_k[self._cur_task].weight.data.copy_(cU[:, :module.rank].T / math.sqrt(3))
                        module.lora_A_v[self._cur_task].weight.data.copy_(cU[:, :module.rank].T / math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
                        kk += 1

        print(f"Parameters to be updated: {enabled}")
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # ------------------------- [MOD] route to FO or ZO -------------------------
        if self.trainer.upper() == "ZO":
            # LAE-style ZO-SPSA training
            self.run_epoch = self.init_epoch if self._cur_task == 0 else self.epochs
            self.train_function_zo(train_loader, test_loader)

        elif self.trainer.upper() == "HYBRID_SPSA":
            self.run_epoch = self.init_epoch if self._cur_task == 0 else self.epochs
            self.train_function_hybrid_spsa(train_loader, test_loader)

        else:
            # Original FO branches unchanged
            if self._cur_task == 0:
                if self.optim == 'sgd':
                    optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,
                                          weight_decay=self.init_weight_decay)
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.init_epoch)
                elif self.optim == 'adam':
                    optimizer = optim.Adam(self._network.parameters(), lr=self.init_lr, weight_decay=self.init_weight_decay,
                                           betas=(0.9, 0.999))
                    scheduler = CosineSchedule(optimizer=optimizer, K=self.init_epoch)
                else:
                    raise Exception
                self.run_epoch = self.init_epoch
                self.train_function(train_loader, test_loader, optimizer, scheduler)
            else:
                if self.optim == 'sgd':
                    optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.lrate,
                                          weight_decay=self.weight_decay)
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.epochs)
                elif self.optim == 'adam':
                    optimizer = optim.Adam(self._network.parameters(), lr=self.lrate, weight_decay=self.weight_decay,
                                           betas=(0.9, 0.999))
                    scheduler = CosineSchedule(optimizer=optimizer, K=self.epochs)
                else:
                    raise Exception
                self.run_epoch = self.epochs
                self.train_function(train_loader, test_loader, optimizer, scheduler)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                self._network(inputs, get_cur_feat=True)

            mat_list = []
            for module in self._network.modules():
                if isinstance(module, Attention_LoRA):
                    mat_list.append(deepcopy(module.cur_matrix))
                    module.cur_matrix.zero_()
                    module.n_cur_matrix = 0
            # self.update_GPM(mat_list)
            self.update_DualGPM(mat_list)

            # Projection Matrix Precomputation
            self.feature_mat = []
            for p in range(len(self.feature_list)):
                Uf = torch.Tensor(np.dot(self.feature_list[p], self.feature_list[p].transpose()))
                print('Layer {} - Projection Matrix shape: {}'.format(p + 1, Uf.shape))
                self.feature_mat.append(Uf)

        return

    # ------------------------- [MOD] helpers for ZO-------------------------



    def _zo_batch_loss(self, x, y):
        """
            InfLoRA-style loss for ZO: only current-task samples, labels reindexed.
        """
        mask = (y >= self._known_classes).nonzero().view(-1)
        if mask.numel() == 0:
            return None
        x_ = torch.index_select(x, 0, mask)
        y_ = torch.index_select(y, 0, mask) - self._known_classes
        with torch.no_grad():
            logits = self._network(x_)['logits']
            return F.cross_entropy(logits, y_.long())
    # -------------------------------------------------------------------------------

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):
            # --------------------------------------------
            # (A) Reset peak memory stats at training start
            # --------------------------------------------
            torch.cuda.reset_peak_memory_stats(device=self._device)

            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes

                logits = self._network(inputs)['logits']
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

                # ---- W&B per-batch ----
                lr_now = optimizer.param_groups[0]["lr"]
                self.global_step += 1
                wandb.log({
                    "global_step": self.global_step,
                    "train/batch_loss": loss.item(),
                    "train/lr": lr_now,
                }, step=self.global_step)

                if self.debug and i > 10: break

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_loss = losses / max(1, len(train_loader))

            peak_mb = 0.0
            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated(self._device) / (1024.0 * 1024.0)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)
            logging.info(info)
            # ---- W&B per-epoch ----
            wandb.log({
                "global_step": self.global_step,
                "epoch": epoch + 1,
                "train/epoch_loss": avg_loss,
                "train/epoch_acc": float(train_acc),
                "mem/peak_allocated_mb": peak_mb,
            }, step=self.global_step)

            logging.info(info)




    # ------------------------- [MOD] LAE-style ZO-SPSA trainer -------------------------
    def train_function_zo(self, train_loader, test_loader):
        """
        Pure zeroth-order SPSA/Q-SPSA, aligned with lae.py:
          - Rademacher (±1) directions
          - Surrogate grads written into p.grad then optimizer step
          - Grad norm clipping
          - Masked logits (default) or current-task-only loss (configurable)
        """
        device = self._device
        epochs = self.run_epoch
        q = int(self.q_spsa)
        eps = float(self.eps)
        max_norm = float(self.max_norm)

        # choose optimizer like LAE
        opt_flag = self.args.get("optimizer", "zo_sgd").lower()
        params = [p for p in self._network.parameters() if p.requires_grad]
        if opt_flag == "zo_sgd":
            opt = optim.SGD(params,
                            lr=self.init_lr if self._cur_task == 0 else self.lrate,
                            momentum=0.9,
                            weight_decay=self.weight_decay)
        elif opt_flag == "zo_adam":
            opt = optim.Adam(params,
                             lr=self.init_lr if self._cur_task == 0 else self.lrate,
                             weight_decay=self.weight_decay)
        else:
            raise ValueError("optimizer must be 'zo_sgd' or 'zo_adam'")

        # scheduler like your FO branches
        if self._cur_task == 0:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.init_epoch)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        self._network.to(device)
        prog_bar = tqdm(range(epochs))

        for ep in prog_bar:
            # --------------------------------------------
            # (A) Reset peak memory stats at training start
            # --------------------------------------------
            torch.cuda.reset_peak_memory_stats(device=self._device)

            # stabilize stochastic layers
            self._network.eval()
            losses = 0.0
            correct, total = 0, 0

            for _, (_, x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)

                # collect trainable params and buffers
                plist = [p for p in self._network.parameters() if p.requires_grad]
                g_buf = [torch.zeros_like(p) for p in plist]

                # Q-SPSA loop with Rademacher ±1
                for _ in range(q):
                    deltas = [torch.randint_like(p, low=0, high=2).mul_(2).sub_(1) for p in plist]

                    # +eps
                    for p, d in zip(plist, deltas):
                        p.data.add_(eps * d)
                    lp = self._zo_batch_loss(x, y)

                    # -eps (net -2eps)
                    for p, d in zip(plist, deltas):
                        p.data.add_(-2 * eps * d)
                    lm = self._zo_batch_loss(x, y)

                    # restore
                    for p, d in zip(plist, deltas):
                        p.data.add_(eps * d)

                    # if batch had no usable samples (current_task mode), skip this direction
                    if lp is None or lm is None:
                        continue

                    diff = (lp - lm)  # scalar tensor
                    for g, d in zip(g_buf, deltas):
                        g.add_(diff * d)

                # write surrogate grads and step (scale = 1/(2 eps q))
                scale = 1.0 / (2.0 * eps * max(1, q))
                has_any_grad = False
                for p, g in zip(plist, g_buf):
                    if g is not None:
                        p.grad = g.mul_(scale)
                        has_any_grad = True

                if not has_any_grad:
                    # no usable directions this batch (rare in current_task mode)
                    continue

                # Zero grads for old FC rows if a monolithic fc exists (parity with LAE)
                if self._cur_task > 0 and hasattr(self._network, "fc"):
                    with torch.no_grad():
                        if getattr(self._network.fc, "weight", None) is not None and self._network.fc.weight.grad is not None:
                            self._network.fc.weight.grad[:self._known_classes].zero_()
                        if getattr(self._network.fc, "bias", None) is not None and self._network.fc.bias.grad is not None:
                            self._network.fc.bias.grad[:self._known_classes].zero_()

                torch.nn.utils.clip_grad_norm_(plist, max_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)

                # stats
                with torch.no_grad():
                    mask = (y >= self._known_classes).nonzero().view(-1)
                    if mask.numel() > 0:
                        x_ = torch.index_select(x, 0, mask)
                        y_ = torch.index_select(y, 0, mask) - self._known_classes
                        logits = self._network(x_)['logits']
                        loss_b = F.cross_entropy(logits, y_.long())
                        preds = logits.argmax(1)
                        correct += preds.eq(y_).sum().item()
                        total += y_.size(0)
                        losses += loss_b.item()

                        # ---- W&B per-batch ----
                        self.global_step += 1
                        wandb.log({
                            "global_step": self.global_step,
                            "train/batch_loss": loss_b.item(),
                            "train/lr": opt.param_groups[0]["lr"],
                        }, step=self.global_step)

            if scheduler:
                scheduler.step()

            train_acc = 100.0 * correct / max(1, total)
            avg_loss = losses / max(1, len(train_loader))

            peak_mb = 0.0
            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated(self._device) / (1024.0 * 1024.0)

            info = (f"[ZO-SPSA] Task {self._cur_task}, Epoch {ep+1}/{epochs} => "
                    f"Loss {losses / max(1, len(train_loader)):.3f}, Train_accy {train_acc:.2f}")
            prog_bar.set_description(info)
            logging.info(info)
            # ---- W&B per-epoch ----
            wandb.log({
                "global_step": self.global_step,
                "epoch": ep + 1,
                "train/epoch_loss": avg_loss,
                "train/epoch_acc": train_acc,
                "mem/peak_allocated_mb": peak_mb,
            }, step=self.global_step)

            logging.info(info)
    # ------------------------------------------------------------------------------------

    def _split_classifier_and_lora_params(self):
        """
        Returns (lora_params, classifier_params) for the *current* task index,
        which is consistently numtask-1 in this codebase.
        """
        lora_params, fc_params = [], []
        numtask = getattr(self._network, "numtask", 1)
        cur_idx = max(0, numtask - 1)

        for name, p in self._network.named_parameters():
            n = name.split("module.", 1)[-1]
            if f"classifier_pool.{cur_idx}" in n:
                fc_params.append(p)
            elif f"lora_B_k.{cur_idx}" in n or f"lora_B_v.{cur_idx}" in n:
                lora_params.append(p)

        return lora_params, fc_params

    def _create_hybrid_optimizers(self, lora_params, fc_params):
        # ZO side optimizer for LoRA params
        if self.hybrid_optimizer == "zo_sgd":
            pet_opt = optim.SGD(lora_params, lr=self.pet_lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.hybrid_optimizer == "zo_adam":
            pet_opt = optim.Adam(lora_params, lr=self.pet_lr, weight_decay=self.weight_decay)
        else:
            raise ValueError("hybrid optimizer must be 'zo_sgd' or 'zo_adam'")

        # FO side optimizer for classifier
        fc_opt = optim.SGD(fc_params, lr=self.fc_lr, momentum=0.9, weight_decay=self.weight_decay)
        return pet_opt, fc_opt

    def _create_hybrid_schedulers(self, pet_opt, fc_opt):
        # mirror your FO schedulers: cosine works well
        if self._cur_task == 0:
            pet_sch = optim.lr_scheduler.CosineAnnealingLR(pet_opt, T_max=self.init_epoch)
            fc_sch = optim.lr_scheduler.CosineAnnealingLR(fc_opt, T_max=self.fc_epoch)
        else:
            pet_sch = optim.lr_scheduler.CosineAnnealingLR(pet_opt, T_max=self.epochs)
            fc_sch = optim.lr_scheduler.CosineAnnealingLR(fc_opt, T_max=self.fc_epoch)
        return pet_sch, fc_sch

    def train_function_hybrid_spsa(self, train_loader, test_loader):
        """
        FO on classifier (current task head), ZO-SPSA on LoRA B_* (current task).
        Loss is computed ONLY on current-task samples (reindexed).
        """
        device = self._device
        epochs = self.run_epoch
        q = int(self.q_spsa)
        eps = float(self.eps)
        max_norm = float(self.max_norm)

        # Split current-task params
        lora_params, fc_params = self._split_classifier_and_lora_params()
        assert len(fc_params) > 0, "No classifier params found for current task."
        assert len(lora_params) > 0, "No LoRA B_k/B_v params found for current task."

        pet_opt, fc_opt = self._create_hybrid_optimizers(lora_params, fc_params)
        pet_sch, fc_sch = self._create_hybrid_schedulers(pet_opt, fc_opt)

        self._network.to(device)
        prog_bar = tqdm(range(epochs))

        for ep in prog_bar:
            # (A) Reset peak memory stat per epoch (optional)
            torch.cuda.reset_peak_memory_stats(device=device)

            # Stability: use eval() as your FO path does (BN/dropout frozen)
            self._network.eval()

            epoch_loss = 0.0
            correct, total = 0, 0

            # For FO-warmup gating
            do_fc_fo = (ep < self.fc_epoch)

            for _, (_, x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)

                # -------- current-task subset --------
                mask = (y >= self._known_classes).nonzero().view(-1)
                if mask.numel() == 0:
                    continue
                x_ = torch.index_select(x, 0, mask)
                y_ = torch.index_select(y, 0, mask) - self._known_classes

                # =======================
                # 1) FO step on classifier
                # =======================
                if do_fc_fo:
                    # Freeze LoRA during FO step
                    for p in lora_params: p.requires_grad_(False)
                    for p in fc_params:   p.requires_grad_(True)

                    fc_opt.zero_grad()
                    logits_fo = self._network(x_)['logits']
                    loss_fo = F.cross_entropy(logits_fo, y_.long())
                    loss_fo.backward()

                    # (optional) clip classifier grads if needed
                    # torch.nn.utils.clip_grad_norm_(fc_params, some_norm)

                    fc_opt.step()

                    # Re-enable LoRA for ZO step
                    for p in lora_params: p.requires_grad_(True)

                # =======================
                # 2) ZO-SPSA step on LoRA
                # =======================
                pet_opt.zero_grad()
                # collect exactly the trainable LoRA params
                zo_params = [p for p in lora_params if p.requires_grad]
                g_buf = [torch.zeros_like(p) for p in zo_params]

                m = 0  # valid SPSA directions actually used
                for _ in range(q):
                    # Rademacher ±1 in the SAME dtype as params
                    deltas = [torch.empty_like(p).bernoulli_(0.5).mul_(2).sub_(1) for p in zo_params]

                    with torch.no_grad():
                        for p, d in zip(zo_params, deltas): p.add_(eps * d)
                        lp = F.cross_entropy(self._network(x_)['logits'], y_.long())

                        for p, d in zip(zo_params, deltas): p.add_(-2 * eps * d)
                        lm = F.cross_entropy(self._network(x_)['logits'], y_.long())

                        for p, d in zip(zo_params, deltas): p.add_(eps * d)

                    diff = (lp - lm)
                    for g, d in zip(g_buf, deltas):
                        g.add_(diff * d)
                    m += 1

                if m > 0:
                    scale = 1.0 / (2.0 * eps * m)
                    for p, g in zip(zo_params, g_buf):
                        p.grad = g.mul_(scale)

                    # clip ZO grads
                    torch.nn.utils.clip_grad_norm_(zo_params, max_norm)
                    pet_opt.step()
                    pet_opt.zero_grad(set_to_none=True)

                # =======================
                # 3) Stats on current task
                # =======================
                with torch.no_grad():
                    logits_now = self._network(x_)['logits']
                    loss_b = F.cross_entropy(logits_now, y_.long())
                    preds = logits_now.argmax(1)
                    correct += preds.eq(y_).sum().item()
                    total += y_.size(0)
                    epoch_loss += loss_b.item()

                    self.global_step += 1
                    wandb.log({
                        "global_step": self.global_step,
                        "train/batch_loss": loss_b.item(),
                        "train/lr_pet": pet_opt.param_groups[0]["lr"],
                        "train/lr_fc": fc_opt.param_groups[0]["lr"],
                    }, step=self.global_step)

            # ---- schedulers per epoch ----
            if pet_sch: pet_sch.step()
            if do_fc_fo and fc_sch: fc_sch.step()

            train_acc = 100.0 * correct / max(1, total)
            avg_loss = epoch_loss / max(1, len(train_loader))

            peak_mb = 0.0
            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

            info = (f"[Hybrid-SPSA] Task {self._cur_task}, Epoch {ep + 1}/{epochs} => "
                    f"Loss {avg_loss:.3f}, Train_acc {train_acc:.2f}, PeakMB {peak_mb:.1f}")
            prog_bar.set_description(info)
            logging.info(info)

            wandb.log({
                "global_step": self.global_step,
                "epoch": ep + 1,
                "train/epoch_loss": avg_loss,
                "train/epoch_acc": train_acc,
                "mem/peak_allocated_mb": peak_mb,
            }, step=self.global_step)

        logging.info("[Hybrid-SPSA] Finished.")



    def clustering(self, dataloader):
        features = []
        for i, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))

    def _evaluate(self, y_pred, y_true):
        ret = {}
        print(len(y_pred), len(y_true))
        grouped = accuracy(y_pred, y_true, self._known_classes, self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []
        y_pred_task, y_true_task = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                y_true_task.append((targets // self.class_num).cpu())

                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs)
                else:
                    outputs = self._network.interface(inputs)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
            y_pred_task.append((predicts // self.class_num).cpu())

            outputs_with_task = torch.zeros_like(outputs)[:, :self.class_num]
            for idx, i in enumerate(targets // self.class_num):
                en, be = self.class_num * i, self.class_num * (i + 1)
                outputs_with_task[idx] = outputs[idx, en:be]
            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + (targets // self.class_num) * self.class_num

            # print(predicts.shape)
            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_true), torch.cat(
            y_pred_task), torch.cat(y_true_task)  # [N, topk]

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def update_DualGPM(self, mat_list):
        threshold = (self.lame - self.lamb) * self._cur_task / self.total_sessions + self.lamb
        print('Threshold: ', threshold)
        if len(self.feature_list) == 0:
            # After First Task
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U, S, Vh = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)  # +1
                if r < (activation.shape[0] / 2):
                    self.feature_list.append(U[:, 0:max(r, 1)])
                    self.project_type.append('remove')
                else:
                    self.feature_list.append(U[:, 0:max(r, 1)])
                    self.project_type.append('retain')
        else:
            for i in range(len(mat_list)):
                if self.project_type[i] == 'remove':
                    activation = mat_list[i]
                    U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1 ** 2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = activation - np.dot(np.dot(self.feature_list[i], self.feature_list[i].transpose()),
                                                  activation)
                    U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S ** 2).sum()
                    sval_ratio = (S ** 2) / sval_total
                    accumulated_sval = (sval_total - sval_hat) / sval_total

                    r = 0
                    for ii in range(sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print('Skip Updating DualGPM for layer: {}'.format(i + 1))
                        continue
                    # update GPM
                    Ui = np.hstack((self.feature_list[i], U[:, 0:r]))
                    if Ui.shape[1] > Ui.shape[0]:
                        self.feature_list[i] = Ui[:, 0:Ui.shape[0]]
                    else:
                        self.feature_list[i] = Ui
                else:
                    assert self.project_type[i] == 'retain'
                    activation = mat_list[i]
                    U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1 ** 2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = np.dot(np.dot(self.feature_list[i], self.feature_list[i].transpose()), activation)
                    U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S ** 2).sum()
                    sval_ratio = (S ** 2) / sval_total
                    accumulated_sval = sval_hat / sval_total

                    r = 0
                    for ii in range(sval_ratio.shape[0]):
                        if accumulated_sval >= (1 - threshold):
                            accumulated_sval -= sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print('Skip Updating DualGPM for layer: {}'.format(i + 1))
                        continue

                    # update GPM by Projected Representation (Eq-8)
                    act_feature = self.feature_list[i] - np.dot(np.dot(U[:, 0:r], U[:, 0:r].transpose()),
                                                                self.feature_list[i])
                    Ui, Si, Vi = np.linalg.svd(act_feature)
                    self.feature_list[i] = Ui[:, :self.feature_list[i].shape[1] - r]

        print('-' * 40)
        print('Gradient Constraints Summary')
        print('-' * 40)
        for i in range(len(self.feature_list)):
            if self.project_type[i] == 'remove' and (
                    self.feature_list[i].shape[1] > (self.feature_list[i].shape[0] / 2)):
                feature = self.feature_list[i]
                # ipdb.set_trace()
                U, S, V = np.linalg.svd(feature)
                new_feature = U[:, feature.shape[1]:]
                self.feature_list[i] = new_feature
                self.project_type[i] = 'retain'
            elif self.project_type[i] == 'retain':
                assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0] / 2)
            print('Layer {} : {}/{} type {}'.format(i + 1, self.feature_list[i].shape[1], self.feature_list[i].shape[0],
                                                    self.project_type[i]))
        print('-' * 40)

    def update_GPM(self, mat_list):
        threshold = (self.lame - self.lamb) * self._cur_task / self.total_sessions + self.lamb
        print('Threshold: ', threshold)
        if len(self.feature_list) == 0:
            # After First Task
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U, S, Vh = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)  # +1
                self.feature_list.append(U[:, 0:max(r, 1)])
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1 ** 2).sum()
                # Projected Representation (Eq-8)
                act_hat = activation - np.dot(np.dot(self.feature_list[i], self.feature_list[i].transpose()),
                                              activation)
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                # criteria (Eq-9)
                sval_hat = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                accumulated_sval = (sval_total - sval_hat) / sval_total

                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    print('Skip Updating GPM for layer: {}'.format(i + 1))
                    continue
                # update GPM
                Ui = np.hstack((self.feature_list[i], U[:, 0:r]))
                if Ui.shape[1] > Ui.shape[0]:
                    self.feature_list[i] = Ui[:, 0:Ui.shape[0]]
                else:
                    self.feature_list[i] = Ui

        print('-' * 40)
        print('Gradient Constraints Summary')
        print('-' * 40)
        for i in range(len(self.feature_list)):
            logging.info('Layer {} : {}/{}'.format(i + 1, self.feature_list[i].shape[1], self.feature_list[i].shape[0]))
        print('-' * 40)