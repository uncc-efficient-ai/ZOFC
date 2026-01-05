import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import LAE
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
        Measured on the *first batch* of each epoch.
        optimizer = grads + optimizer states
        activations = max(forward-only caches, residual at backward/global peak)
        """
        weights_gb = self._gb(weights_b)
        grads_gb = self._gb(grads_b)
        opt_gb = self._gb(opt_b)
        optim_gb = self._gb(grads_b + opt_b)

        activ_fwd_b = max(0, fwd_peak_b - weights_b)
        activ_resid_b = max(0, peak_b - (weights_b + grads_b + opt_b))
        activ_b = max(activ_fwd_b, activ_resid_b)

        activ_fwd_gb = self._gb(activ_fwd_b)
        activ_resid_gb = self._gb(activ_resid_b)
        activ_gb = self._gb(activ_b)
        fwd_peak_gb = self._gb(fwd_peak_b)
        peak_gb = self._gb(peak_b)

        res_peak_b = torch.cuda.max_memory_reserved(self._device) if torch.cuda.is_available() else 0
        overhead_b = max(0, res_peak_b - peak_b)

        # W&B
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

        self._network = LAE(args=args, pretrained=True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.num_freeze_epochs = args["num_freeze_epochs"]
        self.ema_decay = args["ema_decay"]
        for p in self._network.backbone.parameters():
            p.requires_grad_(False)

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

        # SPSA hyper‑params
        self.trainer = args.get("trainer","FO")
        if self.trainer != "FO":
            self.eps      = args["eps"]
            self.q_spsa   = args["q_spsa"]
            self.max_norm = args.get("max_norm", 1.0)

        self.tuned_epoch = args["tuned_epoch"]
        # separate epoch budgets
        if self.trainer=="Hybrid":
            self.tuned_epoch = args["tuned_epoch"]          # total epochs (ZO)
            self.fc_epoch    = args["fc_epoch"]             # epochs with FO on FC

        # bookkeeping
        wandb.init(project="lae_mem", config=args)
        self.global_step = 0

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # snapshot PET params at task start
        self.pet_params_task_start = [p.clone().detach().cpu()
                                      for p in self._network.pets.parameters()]
        if self.trainer == "Hybrid":
            self._train_hybrid()
        else:
            self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        if self.trainer == "ZO":
            self._zeroth_order_train(train_loader, test_loader)
            return

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        if self._cur_task > 0 and self.args["reinit_optimizer"]:
            optimizer = self.get_optimizer()

        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def get_optimizer(self):
        params =  list(self._network.fc.parameters())+list(self._network.pets.parameters())
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                params,
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                params,
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )

        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                params,
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'],
                                                             eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"],
                                                       gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        self._network.attach_pets_vit(self._network.pets)
        for p in self._network.pets.parameters():
            p.requires_grad_(False)
        for _, epoch in enumerate(prog_bar):

            #################################################################
            # (1) Reset peak memory stats at the beginning of the epoch
            #torch.cuda.reset_peak_memory_stats(self._device)
            # reset allocator + ensure no stale grads
            self._reset_peaks()
            optimizer.zero_grad(set_to_none=True)
            self._cuda_sync()

            weights_b = self._bytes_model_weights(self._network)
            measured_this_epoch = False
            #################################################################

            self._network.backbone.train()
            self._network.fc.train()
            if epoch == self.num_freeze_epochs:
                for p in self._network.pets.parameters():
                    p.requires_grad_(True)

            ############# NEW: Snapshot the adapter params #############
            old_params = [p.clone().detach() for p in self._network.pets.parameters()]

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                output = self._network(inputs)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')
                
                loss = F.cross_entropy(logits, targets.long())

                # forward-only peak on first batch
                if not measured_this_epoch and i == 0:
                    self._cuda_sync()
                    fwd_peak_b = torch.cuda.max_memory_allocated(self._device)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

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

                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                
                for param,param_ema in zip(
                self._network.pets.parameters(),self._network.pets_emas.parameters()
            ):
                    param_ema.data = param_ema.data * self.ema_decay + param * (1.0 - self.ema_decay)

                self.global_step += 1

            if scheduler:
                scheduler.step()

            # 3) At epoch's end, measure how much adapter changed
            total_delta = 0.0
            for old, new in zip(old_params, self._network.pets.parameters()):
                # e.g. L2 norm difference
                total_delta += (new.detach() - old).pow(2).sum().item()

            total_delta = total_delta ** 0.5  # sqrt

            avg_loss = losses / len(train_loader)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # ---------------------- Peak Memory ---------------------- #
            peak_memory = torch.cuda.max_memory_allocated(self._device) / (1024.0 * 1024.0)

            if (epoch + 1) == self.args['tuned_epoch']:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = ("Task {}, Epoch {}/{} => "
                        "Adapter Param Δ (L2): {:.6f}, Loss {:.3f}, Train_accy {:.2f}, "
                        "Test_accy {:.2f}, PeakMem {:.2f}MB").format(
                    self._cur_task,
                    epoch + 1, self.args['tuned_epoch'],
                    total_delta, avg_loss, train_acc,
                    test_acc, peak_memory
                )

                # wandb logging for final epoch
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "adapter_param_delta": total_delta,
                    "peak_memory_MB": peak_memory
                }, step=self.global_step)

            else:
                info = ("Task {}, Epoch {}/{} => "
                        "Adapter Param Δ (L2): {:.6f}, Loss {:.3f}, Train_accy {:.2f}, "
                        "PeakMem {:.2f}MB").format(
                    self._cur_task,
                    epoch + 1, self.args['tuned_epoch'],
                    total_delta, avg_loss, train_acc,
                    peak_memory
                )

                # wandb logging for intermediate epochs
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "train_acc": train_acc,
                    "adapter_param_delta": total_delta,
                    "peak_memory_MB": peak_memory
                }, step=self.global_step)
            prog_bar.set_description(info)

        logging.info(info)

    # ----------------------------------------------------------------------
    #                 PURE ZEROTH-ORDER TRAINING (SPSA/Q-SPSA)
    # ----------------------------------------------------------------------
    def _zeroth_order_train(self, train_loader, test_loader):
        device = self._device
        epochs = self.tuned_epoch
        q = self.q_spsa
        eps = self.eps
        max_norm = self.max_norm

        # one optimiser for *all* trainable params
        optim_flag = self.args.get("optimizer", "zo_sgd")
        params = [p for p in self._network.parameters() if p.requires_grad]
        if optim_flag == "zo_sgd":
            opt = optim.SGD(params, lr=self.init_lr, momentum=0.9,
                            weight_decay=self.weight_decay)
        else:
            opt = optim.Adam(params, lr=self.init_lr,
                             weight_decay=self.weight_decay)
        sch = self.get_scheduler(opt)

        self._network.to(device)
        prog_bar = tqdm(range(epochs))

        for ep in prog_bar:
            #torch.cuda.reset_peak_memory_stats(device)

            # reset peaks and grads; get model weight bytes
            self._reset_peaks()
            opt.zero_grad(set_to_none=True)
            self._cuda_sync()
            weights_b = self._bytes_model_weights(self._network)
            measured_this_epoch = False

            self._network.backbone.train(False)
            self._network.fc.train()

            ep_loss, correct, total = 0.0, 0, 0

            for bi, (_, x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)

                # ------------------------------------------------------------------
                # 1.  Collect trainable params & allocate grad buffers
                plist = [p for p in self._network.parameters() if p.requires_grad]
                g_buf = [torch.zeros_like(p) for p in plist]
                total_lp, total_lm = 0.0, 0.0
                # ------------------------------------------------------------------
                # 2.  Q-SPSA loop
                for _ in range(q):
                    deltas = [torch.randint_like(p, 0, 2).mul_(2).sub_(1)  # ±1
                              for p in plist]

                    # +ε
                    for p, d in zip(plist, deltas): p.data.add_(eps * d)
                    with torch.no_grad():
                        lp = F.cross_entropy(self._masked_logits(x), y)

                    # −ε  (net −2ε)
                    for p, d in zip(plist, deltas): p.data.add_(-2 * eps * d)
                    with torch.no_grad():
                        lm = F.cross_entropy(self._masked_logits(x), y)

                    # restore
                    for p, d in zip(plist, deltas): p.data.add_(eps * d)

                    diff = (lp - lm)
                    for g, d in zip(g_buf, deltas):
                        g.add_(diff * d)

                    total_lp += lp.item()
                    total_lm += lm.item()

                # forward-only peak for first batch
                if not measured_this_epoch:
                    self._cuda_sync()
                    fwd_peak_b = torch.cuda.max_memory_allocated(device)

                # ------------------------------------------------------------------
                # 3.  Write surrogate gradients & optimiser step
                scale = 1.0 / (2 * eps * q)
                for p, g in zip(plist, g_buf):
                    p.grad = g.mul_(scale)

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # (3) Zero out old rows so the old classifier cannot update
                if self._cur_task > 0:
                    # e.g. freeze fc.weight[:self._known_classes], fc.bias[:self._known_classes]
                    with torch.no_grad():
                        w_grad = self._network.fc.weight.grad
                        b_grad = self._network.fc.bias.grad
                        if w_grad is not None:
                            w_grad[:self._known_classes].zero_()
                        if b_grad is not None:
                            b_grad[:self._known_classes].zero_()
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                torch.nn.utils.clip_grad_norm_(plist, max_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)

                # global peak + buckets for first batch
                if not measured_this_epoch:
                    self._cuda_sync()
                    peak_b = torch.cuda.max_memory_allocated(device)
                    grads_b = self._bytes_grads(self._network)
                    opt_b = self._bytes_optimizer_states(opt)
                    self._log_mem_buckets(
                        tag=f"mem/ZO_epoch{ep + 1}",
                        weights_b=weights_b, grads_b=grads_b, opt_b=opt_b,
                        fwd_peak_b=fwd_peak_b, peak_b=peak_b
                    )
                    measured_this_epoch = True

                # ------------------------------------------------------------------
                # 4.  Stats on this batch
                with torch.no_grad():
                    logits = self._masked_logits(x)
                    loss_b = F.cross_entropy(logits, y)

                ep_loss += loss_b.item()
                correct += logits.argmax(1).eq(y).sum().item()
                total += y.size(0)

                # EMA for PETs (unchanged from FO code)
                for p, p_ema in zip(self._network.pets.parameters(),
                                    self._network.pets_emas.parameters()):
                    p_ema.data.mul_(self.ema_decay).add_(p.data,
                                                         alpha=1 - self.ema_decay)

                self.global_step += 1
                wandb.log({"train/batch_loss": loss_b.item(),
                           "train/loss_plus": total_lp / q,
                           "train/loss_minus": total_lm / q},
                          step=self.global_step)

            # epoch-level housekeeping --------------------------------------------
            if sch: sch.step()

            peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            tr_loss = ep_loss / len(train_loader)
            tr_acc = 100. * correct / total


            wandb.log({"epoch": ep + 1,
                       "train/epoch_loss": tr_loss,
                       "train/epoch_acc": tr_acc,
                       "peak_memory_gb": peak_mem},
                      step=self.global_step)

            info = (
                f"[ZO] Ep {ep + 1}/{epochs}  loss={tr_loss:.3f} "
                f"train_acc={tr_acc:.1f} "
                f"gpu_mem={peak_mem:.2f}GB")
            logging.info(info)

        logging.info(f"[ZO] Finished {epochs} epochs.")


    # ==================================================================
    # ---------------- Hybrid training loop ----------------------------
    # ==================================================================
    def _train_hybrid(self):
        device = self._device
        epochs = self.tuned_epoch
        q      = self.q_spsa

        pet_params, fc_params = self._split_pet_and_fc()
        pet_opt, fc_opt = self._build_optimizers(pet_params, fc_params)
        pet_sch, fc_sch = self._build_schedulers(pet_opt, fc_opt)

        for p in self._network.pets.parameters():
            p.requires_grad_(True)

        self._network.to(device)
        prog_bar = tqdm(range(epochs))

        for ep in prog_bar:
            torch.cuda.reset_peak_memory_stats(device=device)
            self._network.backbone.train(False)
            self._network.fc.train()

            ep_loss, correct, total = 0.0, 0, 0

            for _, (_, x, y) in enumerate(self.train_loader):
                x, y = x.to(device), y.to(device)

                # ----- (1) FO step on classifier -------------------------
                if ep < self.fc_epoch:
                    for p in self._network.pets.parameters():
                        p.requires_grad_(False)
                    fc_opt.zero_grad(set_to_none=True)
                    loss_fc = F.cross_entropy(self._masked_logits(x), y)
                    loss_fc.backward()
                    fc_opt.step()
                    for p in self._network.pets.parameters():
                        p.requires_grad_(True)

                # ----- (2) ZO‑SPSA step on adapters ---------------------
                pet_opt.zero_grad(set_to_none=True)
                zo_params = [p for p in pet_params if p.requires_grad]
                est_grads = [torch.zeros_like(p) for p in zo_params]

                for _ in range(q):
                    deltas = [torch.randint(0, 2, p.shape, device=p.device,
                                            dtype=p.dtype).mul_(2).sub_(1)
                              for p in zo_params]

                    # +ε
                    for p, d in zip(zo_params, deltas):
                        p.data.add_(self.eps * d)

                    with torch.no_grad():
                        lp = F.cross_entropy(self._masked_logits(x), y)

                    # -ε
                    for p, d in zip(zo_params, deltas):
                        p.data.add_(-2 * self.eps * d)

                    with torch.no_grad():
                        lm = F.cross_entropy(self._masked_logits(x), y)

                    # restore
                    for p, d in zip(zo_params, deltas):
                        p.data.add_(self.eps * d)

                    diff = (lp - lm)
                    for g, d in zip(est_grads, deltas):
                        g.add_(diff * d)

                scale = 1.0 / (2 * self.eps * q)
                for p, g in zip(zo_params, est_grads):
                    p.grad = g.mul_(scale)

                torch.nn.utils.clip_grad_norm_(zo_params, self.max_norm)
                pet_opt.step()

                # ----- (3) stats on this batch --------------------------
                with torch.no_grad():
                    logits = self._masked_logits(x)
                    loss_b = F.cross_entropy(logits, y)

                ep_loss += loss_b.item()
                correct += logits.argmax(1).eq(y).sum().item()
                total   += y.size(0)

                # EMA of PETs
                for p, p_ema in zip(self._network.pets.parameters(),
                                    self._network.pets_emas.parameters()):
                    p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1-self.ema_decay)

                self.global_step += 1
                wandb.log({"batch_loss": loss_b.item()}, step=self.global_step)

            # ---- epoch housekeeping -----------------------------------
            if pet_sch: pet_sch.step()
            if ep < self.fc_epoch and fc_sch: fc_sch.step()

            peak_mem = torch.cuda.max_memory_allocated(device=device) / 1024**3
            pet_delta = self._l2_from_task_start()
            tr_loss   = ep_loss / len(self.train_loader)
            tr_acc    = 100. * correct / total
            te_acc    = self._compute_accuracy(self._network, self.test_loader)

            wandb.log({"epoch": ep+1,
                       "train/epoch_loss": tr_loss,
                       "train/epoch_acc":  tr_acc,
                       "test/acc": te_acc,
                       "PET_param_delta_from_task_start": pet_delta,
                       "peak_memory_gb": peak_mem})

            prog_bar.set_description(
                f"Epoch {ep+1}/{epochs}  loss={tr_loss:.3f}  train_acc={tr_acc:.1f} "
                f"test_acc={te_acc:.1f}  pet_Δ={pet_delta:.3f}  gpu_mem={peak_mem:.2f}GB")

        logging.info(f"[Hybrid] Finished {epochs} epochs.")

    # ==================================================================
    # helpers -----------------------------------------------------------
    # ==================================================================
    def _masked_logits(self, x):
        logits = self._network(x)["logits"][:, :self._total_classes]
        if self._known_classes:
            logits[:, :self._known_classes] = float('-inf')
        return logits

    def _split_pet_and_fc(self):
        pets, fc = [], []
        for n, p in self._network.named_parameters():
            if "fc" in n:
                fc.append(p)
            else:
                pets.append(p)
        return pets, fc

    # -------- newlae‑style optimiser / scheduler helpers --------------
    def _build_optimizers(self, pet_params, fc_params):
        opt_flag = self.args.get("optimizer", "zo_sgd")
        if opt_flag == "zo_sgd":
            pet_opt = optim.SGD(pet_params, lr=self.args["pet_lr"],
                                momentum=0.9, weight_decay=self.weight_decay)
        elif opt_flag == "zo_adam":
            pet_opt = optim.Adam(pet_params, lr=self.args["pet_lr"],
                                 weight_decay=self.weight_decay)
        else:
            raise ValueError("adapter optimiser must be 'zo_sgd' or 'zo_adam'")

        fc_flag = self.args.get("fc_optimizer", "adam").lower()
        if fc_flag == "sgd":
            fc_opt = optim.SGD(fc_params, lr=self.args["fc_lr"],
                               momentum=0.9, weight_decay=self.weight_decay)
        else:
            fc_opt = optim.Adam(fc_params, lr=self.args["fc_lr"],
                                weight_decay=self.weight_decay)
        return pet_opt, fc_opt

    def _build_schedulers(self, pet_opt, fc_opt):
        kind = self.args.get("scheduler", "constant")
        def make(opt, epochs):
            if kind == "cosine":
                return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                            eta_min=self.min_lr)
            elif kind == "steplr":
                return optim.lr_scheduler.MultiStepLR(opt,
                        milestones=self.args["init_milestones"],
                        gamma=self.args["init_lr_decay"])
            return None
        return make(pet_opt, self.tuned_epoch), make(fc_opt, self.fc_epoch)

    def _l2_from_task_start(self):
        tot = 0.0
        for p0, p in zip(self.pet_params_task_start, self._network.pets.parameters()):
            tot += (p.detach().cpu() - p0).pow(2).sum().item()
        return tot ** 0.5

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            output_emas = []
            with torch.no_grad():
                outputs = self._network(inputs)["logits"][:, :self._total_classes]
            output_emas.append(outputs.softmax(dim=1))

            self._network.attach_pets_vit(self._network.pets_emas)
            # using off_model to predict
            with torch.no_grad():
                outputs = self._network(inputs)["logits"][:, :self._total_classes]
            output_emas.append(outputs.softmax(dim=1))

            self._network.attach_pets_vit(self._network.pets)

            outputs = torch.stack(output_emas, dim=-1).max(dim=-1)[0]

            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            pred_on, output_emas = [], []
            #using on_model to predict
            with torch.no_grad():
                outputs = model(inputs)["logits"][:, :self._total_classes]
            output_emas.append(outputs.softmax(dim=1))

            model.attach_pets_vit(model.pets_emas)
            #using off_model to predict
            with torch.no_grad():
                outputs = model(inputs)["logits"][:, :self._total_classes]
            output_emas.append(outputs.softmax(dim=1))

            model.attach_pets_vit(model.pets)

            outputs = torch.stack(output_emas, dim=-1).max(dim=-1)[0]

            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)