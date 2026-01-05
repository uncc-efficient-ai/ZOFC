import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import NewLae
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

from fvcore.nn import FlopCountAnalysis
from utils.flops_handles_custom import apply_custom_jit_handles  # registers the handlers

num_workers = 8

import wandb


class Learner(BaseLearner):

    # ==== FLOPs helpers ====
    def _dim_symbols(self):
        """Return (B,H,W,N,D,C,La,r). Pulls D from network.feature_dim and C from self._total_classes."""
        B = self.batch_size
        H = W = int(self.args.get("image_size", 224))
        # ViT/B16: N = 1 + (H/16)*(W/16)
        N = 1 + (H // 16) * (W // 16)
        D = getattr(self._network, "feature_dim", None)
        if D is None:
            # ViT-B default
            D = 768
        C = int(self._total_classes)
        La = int(self.args.get("pet_La", 5))  # number of blocks with adapters; set this arg or hardcode
        r = int(self.args.get("down_sample_dim", 5))
        return B, H, W, N, D, C, La, r

    def _F_cls_per_forward(self, cache_weight_norm: bool) -> float:
        """
        FLOPs for cosine head per forward:
          features l2-norm: ~3D
          weight l2-norm across C classes: ~3CD (only if not cached)
          class dot products: ~2CD
          scale by sigma: ~C
        """
        _, _, _, _, D, C, _, _ = self._dim_symbols()
        base = (2 * C * D) + (3 * D) + C
        if cache_weight_norm:
            return float(base)  # 3CD paid once/step externally
        else:
            return float(base + 3 * C * D)

    def _F_pet_per_image(self) -> float:
        """Adapter overhead per image: La * N * (4 D r)"""
        _, _, _, N, D, _, La, r = self._dim_symbols()
        return float(La * N * (4 * D * r))

    def _F_fwd_img_theory(self) -> float:
        """
        Total forward FLOPs per image:
          F_backbone(H,W) + F_PET + F_cls
        We approximate F_backbone by measuring it once with fvcore and subtracting cls FLOPs.
        If fvcore isn't available at runtime, we return PET+CLS only (still ok for *relative* deltas).
        """
        cache_weight_norm = bool(self.args.get("cache_unitnorm_weights", False))
        F_cls = self._F_cls_per_forward(cache_weight_norm)
        F_pet = self._F_pet_per_image()
        # Try empirical forward FLOPs once (backbone+adapters+head)
        try:
            if not hasattr(self, "_empirical_F_fwd_img"):
                self._empirical_F_fwd_img = self._empirical_forward_flops_single_image()
            F_total_emp = self._empirical_F_fwd_img
            # Subtract current head FLOPs to approximate backbone part (backbone+PET left in empirical)
            F_backbone_plus_pet_emp = max(0.0, F_total_emp - F_cls)
            # Prefer empirical backbone+PET over our PET-only formula
            return float(F_backbone_plus_pet_emp + F_cls)
        except Exception:
            # Fallback: just PET + CLS (will underestimate absolute FLOPs but preserves deltas between regimes)
            return float(F_pet + F_cls)

    def _empirical_forward_flops_single_image(self) -> float:
        H = W = int(self.args.get("image_size", 224))
        dummy = torch.randn(1, 3, H, W, device=self._device)
        was_training = self._network.training
        self._network.eval()
        with torch.no_grad():
            fca = FlopCountAnalysis(self._network, dummy)
            apply_custom_jit_handles(fca)  # <-- attach handlers here
            f = float(fca.total())
        if was_training:
            self._network.train()
        return f

    def _per_batch_forwards(self, regime: str, epoch: int = 0) -> int:
        """
        How many forward evals per batch are executed in each regime (by your loops):
          FO      : 1
          ZO      : 2q + 1
          HYB(early): 2q + 2  (ZO 2q + one FO + one eval forward)
          HYB(late): 2q + 1  (head frozen; same as ZO)
        """
        q = int(self.args.get("q_spsa", 1))
        if regime == "FO":
            return 1
        elif regime == "ZO":
            return 2 * q + 1
        elif regime == "HYB":
            # early vs late controlled by fc_epoch
            if epoch < int(self.args.get("fc_epoch", 0)):
                return 2 * q + 2
            else:
                return 2 * q + 1
        else:
            return 1

    def _theory_flops_per_batch(self, mode: str, epoch: int):
        B = self.batch_size
        C = self._total_classes
        D = self._network.feature_dim  # 768 for ViT-B
        H = W = 224
        N = 1 + (H // 16) * (W // 16)  # 197
        r = int(self.args.get("down_sample_dim", 5))
        L_a = int(self.args.get("pet_length", 0))  # number of adapter blocks actually inserted

        # 1) Forward costs
        F_pet = float(L_a * N * 4 * D * r)
        # If NOT caching unit-norm weights per step, keep 5CD + 3D + C:
        cache_norms = bool(self.args.get("cache_unitnorm_weights", False))
        if cache_norms:
            F_cls = float(2 * C * D + 3 * D + C)  # 3CD is paid once per step instead
        else:
            F_cls = float(5 * C * D + 3 * D + C)

        per_img_fwd = self._F_fwd_img_theory()  # empirical total forward/img
        per_batch_fwd = B * per_img_fwd
        F_bb = max(0.0, per_img_fwd - (F_pet + F_cls))  # from JIT/profiler once; else 0
        if mode == "FO":
            alpha_fc = float(self.args.get("alpha_fc", 2.0))
            alpha_pet = float(self.args.get("alpha_pet", 2.0))
            alpha_bb = float(self.args.get("alpha_bb", 2.0))  # NEW

            per_batch_total = B * (
                    (1.0 + alpha_bb) * F_bb +
                    (1.0 + alpha_pet) * F_pet +
                    (1.0 + alpha_fc) * F_cls
            )
            forwards_per_batch = 1
        elif mode == "ZO":
            q = int(self.args.get("q_spsa", 1))
            forwards_per_batch = 2 * q + 1
            per_batch_total = forwards_per_batch * per_batch_fwd
        elif mode == "HYB":
            q = int(self.args.get("q_spsa", 1))
            forwards_per_batch = 2 * q + 1
            if epoch < int(self.args.get("fc_epoch", 0)):
                # FO backward only on head (add 2 * F_cls once per batch)
                per_batch_total = forwards_per_batch * per_batch_fwd + B * (2 * F_cls)
            else:
                per_batch_total = forwards_per_batch * per_batch_fwd
        else:
            raise ValueError(mode)

        return {
            "per_image_fwd": per_img_fwd,
            "per_batch_fwd": per_batch_fwd,
            "per_batch_total": per_batch_total,
            "forwards_per_batch": forwards_per_batch,
            "components": {"F_bb": F_bb, "F_pet": F_pet, "F_cls": F_cls}
        }

    def _adapters_are_trainable(self) -> bool:
        try:
            return any(p.requires_grad for p in self._network.pets.parameters())
        except Exception:
            return False

    def _alpha_multipliers(self):
        # ~2× forward is a reasonable rule of thumb for matmul backprop
        alpha_fc = float(self.args.get("alpha_fc", 2.0))
        alpha_pet = float(self.args.get("alpha_pet", 2.0))
        return alpha_fc, alpha_pet
    # ==== end FLOPs helpers ====

    def __init__(self, args):
        super().__init__(args)

        self._network = NewLae(args=args, pretrained=True)

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

        # zo eps
        if args["optimizer"] == "zo_sgd" or args["optimizer"] == "zo_adam":
            self.eps = args["eps"]
            self.q_spsa = args["q_spsa"]
            print("eps:", self.eps)
            print("q_spsa:", self.q_spsa)

        # Flatness measurement settings from args
        self.rho_0th   = args.get("rho_unified", 1e-3)
        self.rho_1st   = args.get("rho_first",   1e-3)
        self.n_dir     = args.get("n_directions_unified", 100)


        wandb.init(project="newlae_domainnet", config=args)
        self.global_step = 0

    def after_task(self):
        self._known_classes = self._total_classes

        with torch.no_grad():
            self._network.fc.weight[:self._known_classes].requires_grad_(False)

        '''
        self._log_classifier_weight_norms()
        
        # Evaluate on old tasks
        old_tasks_loader = self._build_old_tasks_loader()

        if old_tasks_loader is not None:

            L0 = self._evaluate_loss(old_tasks_loader)   # base loss 𝓛₀
            logging.info("[Flatness] Base old‑task loss L0 = %.6e", L0)

            # ============ 0‑th order unified (L∞ and L2) ============
            z_u_stats = {}
            for norm in ("linf", "l2"):
                stats = self._zero_order_unified(old_tasks_loader, rho=self.rho_0th, n_dir=self.n_dir, norm=norm)
                tag = f"zoU_{norm}"
                z_u_stats[tag] = stats  # returns dict

            # ============ 0‑th order SAM ============================
            zo_sam_loss = self._zero_order_sam_loss(old_tasks_loader, rho=self.rho_0th)

            # ============ First‑order unified (gradient‑oriented) ===
            f_u_stats = {}
            for norm in ("linf", "l2"):
                stats = self._first_order_unified(old_tasks_loader, rho=self.rho_1st, n_dir=self.n_dir, norm=norm)
                f_u_stats[f"foU_{norm}"] = stats

            # ============ First‑order SAM gradient change ===========
            fo_sam_grad = self._first_order_sam_grad(old_tasks_loader, rho=self.rho_1st)

            # ------------ logging -----------------------------------
            zo_sam_loss_norm = zo_sam_loss / (L0 + 1e-12)
            fo_sam_grad_norm = fo_sam_grad / (L0 + 1e-12)

            log_dict = {"L0": L0,
                        "zo_sam_loss": zo_sam_loss,
                        "zo_sam_loss_norm": zo_sam_loss_norm,
                        "fo_sam_grad": fo_sam_grad,
                        "fo_sam_grad_norm": fo_sam_grad_norm}

            # expand the two dicts
            for k, v in z_u_stats.items():
                for kk, vv in v.items(): log_dict[f"{k}_{kk}"] = vv
            for k, v in f_u_stats.items():
                for kk, vv in v.items(): log_dict[f"{k}_{kk}"] = vv
            wandb.log(log_dict, step=self.global_step)

            # print every result key→value pair
            logging.info("─ Flatness results (task %s) ─", self._cur_task)
            for k in sorted(log_dict.keys()):
                logging.info("    %-28s : %.6e", k, log_dict[k])
            logging.info("────────────────────────────────────────")

            # sam summary
            logging.info("[Flatness] ZO‑SAM‑loss = %.4e – FO‑SAM‑grad = %.4e",
                         zo_sam_loss, fo_sam_grad)
        '''
        if "num_tasks" in self.args:
            # If this is the final task, optionally save final model
            if self._cur_task == self.args["num_tasks"] - 1:
                self.save_final_model()


    def _log_classifier_weight_norms(self):
        """
        Log the average classifier weight norm *per task*.
        E.g. at the end of Task k, we log the average norm for tasks 0..k,
        each of which has a known block of classes in the FC layer.
        """
        fc_weights = self._network.fc.weight.data  # shape [total_classes, feat_dim]
        weight_norms = fc_weights.norm(dim=1)  # shape [total_classes]

        # We'll iterate through all tasks from 0.._cur_task
        # and compute the average norm for each task's subset of classes.
        offset = 0
        for task_id in range(self._cur_task + 1):
            task_size = self.data_manager.get_task_size(task_id)
            # The classes for `task_id` go from offset..offset+task_size-1
            subset_norms = weight_norms[offset: offset + task_size]
            offset += task_size

            # Average norm across these classes
            avg_norm = subset_norms.mean().item()

            # Log one value for each task
            # e.g. "Task_0_avg_classifier_weight_norm", "Task_1_avg_classifier_weight_norm", ...
            wandb.log({
                f"Task_{task_id}_avg_classifier_weight_norm": avg_norm,
                "logged_at_task": self._cur_task,
            }, step=self.global_step)

            # 2) Also print / log it to console
            logging.info(
                f"[WeightNorm] Task {task_id}: Avg classifier weight norm = {avg_norm:.4f}"
            )

        logging.info(
            f"Logged average classifier weight norms (Task 0..{self._cur_task}) at the end of Task {self._cur_task}."
        )

    def save_final_model(self, save_path=None):
        """
        Save the current model's state_dict to disk.

        If no save_path is provided, we look for self.args["save_path"].
        If that doesn't exist either, default to "final_model.pth".
        """
        if save_path is None:
            save_path = self.args.get("save_path", "final_model.pth")

        model_to_save = self._network
        if isinstance(model_to_save, nn.DataParallel):
            model_to_save = model_to_save.module

        torch.save(model_to_save.state_dict(), save_path)
        logging.info(f"Final model saved to {save_path}")


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

        # (1) TAKE A SNAPSHOT for PET parameters now, “task start”
        self.pet_params_task_start = [
            p.clone().detach().cpu() for p in self._network.pets.parameters()
        ]

        # If you also want to monitor the FC:
        #self.fc_params_task_start = [
        #    p.clone().detach() for p in self._network.fc.parameters()
        #]


        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module



    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        # Hybrid SPSA path
        if self.args['trainer'] == 'hybrid_spsa':
            # Retrieve per‐group learning rates, or default to init_lr
            pet_lr = self.args.get("pet_lr", self.init_lr)  # for adapters
            fc_lr = self.args.get("fc_lr", self.init_lr)  # for classifier
            self._train_hybrid_spsa(train_loader, test_loader, pet_lr, fc_lr)
            return
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        if self._cur_task > 0 and self.args["reinit_optimizer"]:
            optimizer = self.get_optimizer()
            scheduler = self.get_scheduler(optimizer)

        # STEP 1: Decide if we do FO or ZO
        if self.args['optimizer'] == 'zo_adam' or self.args['optimizer'] == 'zo_sgd':
            # Zeroth-order branch
            self._zeroth_order_train(train_loader, test_loader, optimizer, scheduler)
        else:
            # First-order (default) branch
            self._init_train(train_loader, test_loader, optimizer, scheduler)

    def get_optimizer(self):
        params = list(self._network.fc.parameters()) + list(self._network.pets.parameters())
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

        elif self.args['optimizer'] == 'zo_sgd':
            optimizer = optim.SGD(
                params,
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )

        elif self.args['optimizer'] == 'zo_adam':
            optimizer = optim.Adam(
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
        # for name, param in self._network.named_parameters():
        #    if 'adapter' in name:
        #        print(f"Found adapter: {name}")
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        self._network.attach_pets_vit(self._network.pets)
        for p in self._network.pets.parameters():
            p.requires_grad_(False)
        for _, epoch in enumerate(prog_bar):
            # 1) Reset peak memory stats at start of epoch
            torch.cuda.reset_peak_memory_stats(device=self._device)
            self._network.backbone.train()
            self._network.fc.train()

            if epoch == self.num_freeze_epochs:
                for p in self._network.pets.parameters():
                    p.requires_grad_(True)

            # --- FLOPs logging (FO) ---
            if self.args.get("measure_flops", False):
                stats = self._theory_flops_per_batch("FO", epoch)
                wandb.log({
                    "flops/FO/per_image_forward": stats["per_image_fwd"],
                    "flops/FO/per_batch_forward": stats["per_batch_fwd"],
                    "flops/FO/per_batch_total": stats["per_batch_total"],
                    "flops/FO/forwards_per_batch": stats["forwards_per_batch"],
                    "flops/components/F_bb": stats["components"]["F_bb"],
                    "flops/components/F_pet": stats["components"]["F_pet"],
                    "flops/components/F_cls": stats["components"]["F_cls"],
                }, step=self.global_step)
                logging.info(
                    "[FLOPs][FO] epoch=%d fwd/img=%.3e, fwd/batch=%.3e, total/batch=%.3e",
                    epoch, stats["per_image_fwd"], stats["per_batch_fwd"], stats["per_batch_total"]
                )

            ############# NEW: Snapshot the adapter params #############
            #old_params = [p.clone().detach() for p in self._network.pets.parameters()]

            epoch_loss = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                output = self._network(inputs)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')

                loss = F.cross_entropy(logits, targets.long())
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                epoch_loss += loss.item()

                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

                self.global_step += 1  # increment the global step
                wandb.log({
                    "batch_loss": loss.item()
                }, step=self.global_step)

            #    for param,param_ema in zip(
            #    self._network.pets.parameters(),self._network.pets_emas.parameters()
            # ):
            #        param_ema.data = param_ema.data * self.ema_decay + param * (1.0 - self.ema_decay)

            if scheduler:
                scheduler.step()

            # 3) Measure peak memory usage for this epoch
            peak_mem_bytes = torch.cuda.max_memory_allocated(device=self._device)
            peak_mem_gb = peak_mem_bytes / (1024 ** 3)

            # 3) At epoch's end, measure how much adapter changed
            #total_delta = 0.0
            #for old, new in zip(old_params, self._network.pets.parameters()):
            #    # e.g. L2 norm difference
            #    total_delta += (new.detach() - old).pow(2).sum().item()

            #total_delta = total_delta ** 0.5  # sqrt
            #task_delta = 0.0
            #for param_start, param_now in zip(self.pet_params_task_start, self._network.pets.parameters()):
            #    diff = (param_now.detach().cpu() - param_start).pow(2).sum()
            #    task_delta += diff.item()

            #task_delta = task_delta ** 0.5  # L2 norm


            train_loss = epoch_loss / len(train_loader)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)


            # Log epoch-level stats to W&B
            wandb.log({
                "epoch": epoch + 1,
                "train/epoch_loss": train_loss,
                "train/epoch_acc": train_acc,
                "peak_memory_gb": peak_mem_gb
            }, step=self.global_step)

            logging.info(
                f"[FO] Task {self._cur_task}, Epoch {epoch + 1}/{self.args['tuned_epoch']} => "
                f"Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}, "
                f"Peak GPU Mem: {peak_mem_gb:.2f} GB"
            )

        logging.info(f"[FO] Finished training after {self.args['tuned_epoch']} epochs.")

    def _zeroth_order_train(self, train_loader, test_loader, optimizer, scheduler):
        """
        Minimal-memory 2-point (SPSA/Q-SPSA) training, adapted from newease.py.
        """

        # Decide how many epochs to train
        epochs = self.args['tuned_epoch']

        # Q-SPSA “q_spsa” - number of repeated perturbations per mini-batch
        q_spsa = self.args.get("q_spsa", 1)

        prog_bar = tqdm(range(epochs))
        self._network.to(self._device)

        for _, epoch in enumerate(prog_bar):
            # --------------------------------------------
            # (A) Reset peak memory stats at training start
            # --------------------------------------------
            torch.cuda.reset_peak_memory_stats(device=self._device)

            self._network.train()
            epoch_loss = 0.0
            correct, total = 0, 0

            # --- FLOPs logging (ZO) ---
            if self.args.get("measure_flops", False):
                stats = self._theory_flops_per_batch("ZO", epoch)
                wandb.log({
                    "flops/ZO/per_image_forward": stats["per_image_fwd"],
                    "flops/ZO/per_batch_forward": stats["per_batch_fwd"],
                    "flops/ZO/per_batch_total": stats["per_batch_total"],  # = forward only
                    "flops/ZO/forwards_per_batch": self._per_batch_forwards("ZO", epoch),  # = 2q+1
                    "flops/ZO/q": int(self.args.get("q_spsa", 1)),
                }, step=self.global_step)
                logging.info("[FLOPs][ZO] epoch=%d fwd/img=%.3e, fwd/batch=%.3e, total/batch=%.3e, q=%d",
                             epoch, stats["per_image_fwd"], stats["per_batch_fwd"], stats["per_batch_total"],
                             int(self.args.get("q_spssa", self.args.get("q_spsa", 1))))

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # Collect trainable parameters
                param_list = [p for p in self._network.parameters() if p.requires_grad]
                # Accumulate approximate gradients here
                param_grads = [torch.zeros_like(p.data) for p in param_list]

                # For logging the sum of loss_plus and loss_minus
                total_loss_plus = 0.0
                total_loss_minus = 0.0

                # Main Q-SPSA loop
                for _ in range(q_spsa):
                    # 1) Sample Rademacher deltas
                    deltas = []
                    for p in param_list:
                        delta = torch.randint(low=0, high=2, size=p.shape, device=self._device, dtype=p.dtype)
                        # Convert {0,1} to {-1,+1}
                        delta = delta * 2 - 1
                        deltas.append(delta)

                    # 2) Evaluate loss at (theta + eps*delta)
                    #    SHIFT params by +eps
                    for p, d in zip(param_list, deltas):
                        p.data.add_(self.eps * d)

                    with torch.no_grad():
                        logits_plus = self._network(inputs)["logits"][:, :self._total_classes]
                        # LAE style masking
                        logits_plus[:, :self._known_classes] = float('-inf')
                        loss_plus = F.cross_entropy(logits_plus, targets.long())

                    total_loss_plus += loss_plus.item()

                    # 3) Evaluate loss at (theta - eps*delta)
                    #    SHIFT params from +eps*d to -eps*d => net -2*eps*d
                    for p, d in zip(param_list, deltas):
                        p.data.add_(-2.0 * self.eps * d)

                    with torch.no_grad():
                        logits_minus = self._network(inputs)["logits"][:, :self._total_classes]
                        logits_minus[:, :self._known_classes] = float('-inf')
                        loss_minus = F.cross_entropy(logits_minus, targets.long())

                    total_loss_minus += loss_minus.item()

                    # 4) Restore original parameters => add +eps*d
                    for p, d in zip(param_list, deltas):
                        p.data.add_(self.eps * d)

                    # 5) Accumulate approximate gradient
                    diff = (loss_plus - loss_minus)
                    for idx, p in enumerate(param_list):
                        param_grads[idx].add_(diff * deltas[idx])

                # Now scale the accumulated gradient by 1/(2*eps*q_spsa)
                scale_factor = 1.0 / (2.0 * self.eps * q_spsa)
                for idx, p in enumerate(param_list):
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    p.grad.detach_()
                    p.grad.copy_(param_grads[idx] * scale_factor)

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # (3) Zero out old rows so the old classifier cannot update
                if self._cur_task > 0:
                    # e.g. freeze fc.weight[:self._known_classes], fc.bias[:self._known_classes]
                    with torch.no_grad():
                        w_grad = self._network.fc.weight.grad
                        #b_grad = self._network.fc.bias.grad
                        if w_grad is not None:
                            w_grad[:self._known_classes].zero_()
                        #if b_grad is not None:
                        #    b_grad[:self._known_classes].zero_()
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                # 2) GRAD CLIPPING step here
                # E.g., clip all trainable params by max_norm=1.0 (or whatever you pick)
                torch.nn.utils.clip_grad_norm_(param_list, max_norm=self.args.get("max_norm", 1.0))


                # 6) Standard optimizer step => updates theta
                optimizer.step()

                optimizer.zero_grad()

                # Evaluate updated model on the same mini-batch
                with torch.no_grad():
                    logits = self._network(inputs)["logits"][:, :self._total_classes]
                    logits[:, :self._known_classes] = float('-inf')
                    loss_batch = F.cross_entropy(logits, targets.long())

                epoch_loss += loss_batch.item()
                # Accuracy vs. masked aux_targets
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

                self.global_step += 1
                # Log batch-level info
                wandb.log({
                    "train/batch_loss": loss_batch.item(),
                    "train/loss_plus_avg": total_loss_plus / q_spsa,
                    "train/loss_minus_avg": total_loss_minus / q_spsa
                }, step=self.global_step)

            # END of all mini-batches in this epoch

            # Optional scheduler
            if scheduler:
                scheduler.step()

            # ------------------------------------------------------
            # (B) Check the peak GPU memory usage after this epoch
            # ------------------------------------------------------
            peak_mem_bytes = torch.cuda.max_memory_allocated(device=self._device)
            peak_mem_gb = peak_mem_bytes / (1024 ** 3)
            logging.info(
                f"[ZO-SPSA] Epoch {epoch + 1}/{epochs} => "
                f"Peak GPU memory usage so far: {peak_mem_gb:.2f} GB"
            )

            train_loss = epoch_loss / len(train_loader)
            train_acc = 100.0 * correct / total if total > 0 else 0.0


            # Log epoch-level stats
            wandb.log({
                "epoch": epoch + 1,
                "train/epoch_loss": train_loss,
                "train/epoch_acc": train_acc,
                "peak_memory_gb": peak_mem_gb
            }, step=self.global_step)

            # Evaluate on test data
            info = (f"[ZO-SPSA] Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Train Loss {train_loss:.3f}, Train Acc {train_acc:.2f}")
            logging.info(info)

        logging.info(f"[ZO] Finished training after {self.args['tuned_epoch']} epochs.")

    def _split_backbone_and_fc_parameters(self):
        """
        Return two lists of parameters:
          - pet_params: e.g. self._network.pets
          - fc_params:      self._network.fc
        """
        pet_params = []
        fc_params = []
        for name, param in self._network.named_parameters():
            # In newLAE, "pets" is the adapter
            if "pet" in name or "prefix" in name or "adapter" in name or "lora" in name:
                pet_params.append(param)
            elif "fc" in name:
                fc_params.append(param)
            # The backbone is already frozen in __init__, so we ignore it
        return pet_params, fc_params

    def _create_hybrid_optimizers(self, pet_params, fc_params, pet_lr, fc_lr):
        """
        Create two separate optimizers for the adapter/backbone (SPSA)
        and the classifier (FO). We still use e.g. PyTorch SGD or Adam,
        but we will manually assign .grad for SPSA.
        """
        if self.args['optimizer'] == 'zo_sgd':
            pet_optimizer = optim.SGD(
                pet_params,
                lr=pet_lr,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'zo_adam':
            pet_optimizer = optim.Adam(
                pet_params,
                lr=pet_lr,
                weight_decay=self.weight_decay
            )

        fc_optimizer = optim.SGD(
            fc_params,
            lr=fc_lr,
            momentum=0.9,
            weight_decay=self.weight_decay
        )

        '''
        fc_optimizer = optim.Adam(
            fc_params,
            lr=fc_lr,
            weight_decay=0 #self.weight_decay
        )
        '''
        #for i, group in enumerate(pet_optimizer.param_groups):
        #    print(f"Group {i}, LR={group['lr']}")
        #    for p in group['params']:
        #        print("   ", p.shape, p.requires_grad, id(p))

        return pet_optimizer, fc_optimizer

    def _create_hybrid_schedulers(self, pet_optimizer, fc_optimizer, total_epochs):
        """
        Optionally create separate schedulers for each optimizer, e.g. Cosine or StepLR.
        """
        if self.args["scheduler"] == 'cosine':
            pet_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                pet_optimizer, T_max=total_epochs, eta_min=self.min_lr
            )
            fc_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                fc_optimizer, T_max=self.args["fc_epoch"], eta_min=self.min_lr
            )
            #fc_scheduler = None
        elif self.args["scheduler"] == 'steplr':
            pet_scheduler = optim.lr_scheduler.MultiStepLR(
                pet_optimizer,
                milestones=self.args["init_milestones"],
                gamma=self.args["init_lr_decay"]
            )
            fc_scheduler = optim.lr_scheduler.MultiStepLR(
                fc_optimizer,
                milestones=self.args["init_milestones"],
                gamma=self.args["init_lr_decay"]
            )
        else:
            pet_scheduler = None
            #fc_scheduler = None
            fc_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                fc_optimizer, T_max=self.args["fc_epoch"], eta_min=self.min_lr
            )

        return pet_scheduler, fc_scheduler

    def _freeze_classifier(self):
        """
        Freeze the classifier so no more FO updates.
        """
        for p in self._network.fc.parameters():
            p.requires_grad = False

    def _train_hybrid_spsa(self, train_loader, test_loader, pet_lr, fc_lr):
        #for name, param in self._network.named_parameters():
        #    if 'adapter' in name:
        #        print(f"Found adapter: {name}")
        epochs = self.args['tuned_epoch']
        q_spsa = self.q_spsa
        self._network.attach_pets_vit(self._network.pets)
        for p in self._network.pets.parameters():
            p.requires_grad_(True)

        #logging.info("Listing all named_parameters in the network:")
        #for name, param in self._network.named_parameters():
        #    logging.info(f"Param: {name}, shape: {param.shape}, requires_grad={param.requires_grad}")

        # Split parameters
        pet_params, fc_params = self._split_backbone_and_fc_parameters()



        # -------------------------------------------------------------------
        # Display the total and trainable param counts for PET parameters
        #pet_total_params = sum(p.numel() for p in pet_params)
        #pet_trainable_params = sum(p.numel() for p in pet_params if p.requires_grad)
        #logging.info(f"[Hybrid-SPSA] PET parameter count: total={pet_total_params}, trainable={pet_trainable_params}")
        # -------------------------------------------------------------------

        # Create two optimizers (and schedulers if needed)
        pet_optimizer, fc_optimizer = self._create_hybrid_optimizers(pet_params, fc_params, pet_lr, fc_lr)
        pet_scheduler, fc_scheduler = self._create_hybrid_schedulers(pet_optimizer, fc_optimizer, epochs)

        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            # --- (1) Snapshot old adapter params at start of epoch ---
            #old_params = [p.clone().detach() for p in self._network.pets.parameters()]

            #
            # (A) Reset peak memory usage stats at start of epoch
            #
            torch.cuda.reset_peak_memory_stats(device=self._device)

            self._network.backbone.train()
            self._network.fc.train()

            # --- FLOPs logging (HYB) ---
            if self.args.get("measure_flops", False):
                stats = self._theory_flops_per_batch("HYB", epoch)
                wandb.log({
                    "flops/HYB/per_image_forward": stats["per_image_fwd"],
                    "flops/HYB/per_batch_forward": stats["per_batch_fwd"],
                    "flops/HYB/per_batch_total": stats["per_batch_total"],
                    "flops/HYB/forwards_per_batch": self._per_batch_forwards("HYB", epoch),  # early: 2q+2; late: 2q+1
                    "flops/HYB/q": int(self.q_spsa),
                    "flops/HYB/is_early": int(epoch < int(self.args.get("fc_epoch", 0))),
                }, step=self.global_step)
                logging.info("[FLOPs][HYB] epoch=%d fwd/img=%.3e, fwd/batch=%.3e, total/batch=%.3e, early=%s, q=%d",
                             epoch, stats["per_image_fwd"], stats["per_batch_fwd"], stats["per_batch_total"],
                             epoch < int(self.args.get("fc_epoch", 0)), int(self.q_spsa))

            # If we've reached freeze_epoch, freeze classifier
            #if epoch == 20:
            #    self._freeze_classifier()

            epoch_loss = 0.0
            correct, total = 0, 0

            for bidx, (img_ids, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # FO step on classifier (if not frozen)
                # If fc_params are still requires_grad => do standard backprop
                # -------- FO pass on classifier --------
                #fwd_peak_fo_b = 0

                if epoch<self.args['fc_epoch'] and any(pp.requires_grad for pp in fc_params):
                    for p in self._network.pets.parameters():
                        p.requires_grad_(False)
                    fc_optimizer.zero_grad()
                    logits_fo = self._network(inputs)["logits"][:, :self._total_classes]
                    logits_fo[:, :self._known_classes] = float('-inf')
                    loss_fo = F.cross_entropy(logits_fo, targets.long())
                    loss_fo.backward()
                    fc_optimizer.step()
                    for p in self._network.pets.parameters():
                        p.requires_grad_(True)


                # ZO step on pets
                # accumulate approximate gradient in memory for pet_params only
                pet_optimizer.zero_grad()
                zo_params = [p for p in pet_params if p.requires_grad]
                zo_grads = [torch.zeros_like(p.data) for p in zo_params]

                total_loss_plus, total_loss_minus = 0.0, 0.0

                for _ in range(q_spsa):
                    # Sample Rademacher deltas
                    deltas = []
                    for p in zo_params:
                        delta = torch.randint(0, 2, size=p.shape, device=p.device, dtype=p.dtype)
                        delta = delta * 2 - 1  # {0,1} => {-1,+1}
                        deltas.append(delta)

                    # SHIFT +eps
                    for p, d in zip(zo_params, deltas):
                        p.data.add_(self.eps * d)

                    with torch.no_grad():
                        logits_plus = self._network(inputs)["logits"][:, :self._total_classes]
                        logits_plus[:, :self._known_classes] = float('-inf')
                        loss_plus = F.cross_entropy(logits_plus, targets.long())

                    total_loss_plus += loss_plus.item()

                    # SHIFT -eps => net -2*eps
                    for p, d in zip(zo_params, deltas):
                        p.data.add_(-2.0 * self.eps * d)

                    with torch.no_grad():
                        logits_minus = self._network(inputs)["logits"][:, :self._total_classes]
                        logits_minus[:, :self._known_classes] = float('-inf')
                        loss_minus = F.cross_entropy(logits_minus, targets.long())

                    total_loss_minus += loss_minus.item()

                    # Restore
                    for p, d in zip(zo_params, deltas):
                        p.data.add_(self.eps * d)

                    # Accumulate approx gradient
                    diff = (loss_plus - loss_minus)
                    for idx_z, p in enumerate(zo_params):
                        zo_grads[idx_z].add_(diff * deltas[idx_z])

                # Scale the accumulated gradient
                scale_factor = 1.0 / (2.0 * self.eps * q_spsa)
                for idx_z, p in enumerate(zo_params):
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    p.grad.detach_()
                    p.grad.copy_(zo_grads[idx_z] * scale_factor)

                # (2) CLIP adapter grads
                max_norm = self.args.get("max_norm", 1.0)
                torch.nn.utils.clip_grad_norm_(zo_params, max_norm=max_norm)


                # Update pets
                pet_optimizer.step()

                '''
                # FO step on classifier (if not frozen)
                # If fc_params are still requires_grad => do standard backprop
                if any(pp.requires_grad for pp in fc_params):
                    fc_optimizer.zero_grad()
                    logits_fo = self._network(inputs)["logits"][:, :self._total_classes]
                    logits_fo[:, :self._known_classes] = float('-inf')
                    loss_fo = F.cross_entropy(logits_fo, targets.long())
                    loss_fo.backward()
                    fc_optimizer.step()
                '''

                # Evaluate on the same batch with updated model
                with torch.no_grad():
                    logits_final = self._network(inputs)["logits"][:, :self._total_classes]
                    logits_final[:, :self._known_classes] = float('-inf')
                    loss_batch = F.cross_entropy(logits_final, targets.long())

                epoch_loss += loss_batch.item()
                _, preds = torch.max(logits_final, dim=1)
                correct += (preds == targets).sum().item()
                total += len(targets)

                self.global_step += 1
                wandb.log({
                    "train/batch_loss": loss_batch.item(),
                    "train/loss_plus_avg": total_loss_plus / q_spsa,
                    "train/loss_minus_avg": total_loss_minus / q_spsa
                }, step=self.global_step)

            # End of epoch
            if pet_scheduler:
                pet_scheduler.step()
            if epoch<self.args['fc_epoch'] and fc_scheduler and any(p.requires_grad for p in fc_params):
                fc_scheduler.step()

            #
            # (C) Measure peak memory usage at end of epoch
            #
            peak_mem_bytes = torch.cuda.max_memory_allocated(device=self._device)
            peak_mem_gb = peak_mem_bytes / (1024 ** 3)

            # --- (2) Compute total adapter param delta (L2) at epoch's end ---
            #total_delta = 0.0
            #for old, new in zip(old_params, self._network.pets.parameters()):
            #    total_delta += (new.detach() - old).pow(2).sum().item()
            #total_delta = total_delta ** 0.5

            # (A) Now compute the distance from the task start
            pet_task_delta = 0.0
            for param_start, param_now in zip(self.pet_params_task_start, self._network.pets.parameters()):
                diff = (param_now.detach().cpu() - param_start).pow(2).sum()
                pet_task_delta += diff.item()
            pet_task_delta = pet_task_delta ** 0.5

            train_loss = epoch_loss / len(train_loader)
            train_acc = 100.0 * correct / total if total > 0 else 0.0

            # wandb log
            wandb.log({
                "epoch": epoch + 1,
                "PET_param_delta_from_task_start": pet_task_delta,
                "train/epoch_loss": train_loss,
                "train/epoch_acc": train_acc,
                "peak_memory_gb": peak_mem_gb
            }, step=self.global_step)

            info = (f"[Hybrid-SPSA] Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"PET Param Δ (L2): {pet_task_delta:.6f}, "
                    f"Train Loss {train_loss:.3f}, Train Acc {train_acc:.2f}, "
                    f"Peak Mem {peak_mem_gb:.2f} GB"
                    )
            logging.info(info)

        logging.info(f"[Hybrid-SPSA] Finished training after {epochs} epochs.")

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"][:, :self._total_classes]

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
            # using on_model to predict
            with torch.no_grad():
                outputs = model(inputs)["logits"][:, :self._total_classes]

            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_test_loss_and_acc(self, model, loader):
        """
        Evaluate average cross-entropy loss and accuracy on a given DataLoader.
        Returns: (test_loss, test_acc)
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                with torch.no_grad():
                    outputs = model(inputs)["logits"][:, :self._total_classes]

                    loss = F.cross_entropy(outputs, targets.long())

                total_loss += loss.item() * len(targets)

                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts == targets).sum()
                total += len(targets)

        avg_loss = total_loss / total if total > 0 else 0.0
        avg_acc = 100.0 * correct / total if total > 0 else 0.0
        return avg_loss, avg_acc


    # ------------------------------------------------------------------------
    # OLD TASK LOADER & EVAL
    # ------------------------------------------------------------------------
    def _build_old_tasks_loader(self):
        if self._cur_task == 0:
            return None
        old_class_range = np.arange(0, self._known_classes)
        old_dataset = self.data_manager.get_dataset(indices=old_class_range, source="test", mode="test")
        loader = DataLoader(old_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        return loader

    def _evaluate_loss(self, loader):
        self._network.eval()
        total_loss, count = 0.0, 0
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out = self._network(inputs)["logits"][:, :self._total_classes]
                loss = F.cross_entropy(out, targets.long())
                total_loss += loss.item() * len(targets)
                count += len(targets)
        return total_loss / count if count > 0 else 0.0

    def _compute_old_tasks_loss_and_grad(self, loader):
        """Computes loss and gradients over the old tasks."""
        for p in self._network.parameters():
            if p.grad is not None:
                p.grad.zero_()

        self._network.train()  # Enable train mode for gradients
        total_loss, total_count = 0.0, 0

        # We compute gradients over the full dataset for stability
        for _, inputs, targets in loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            out = self._network(inputs)["logits"][:, :self._total_classes]

            # Scale loss by batch size so that the sum of gradients is correct
            loss = F.cross_entropy(out, targets.long())

            # Backpropagate to accumulate gradients
            loss.backward()

            total_loss += loss.item() * len(targets)
            total_count += len(targets)

        self._network.eval()  # Return to eval mode

        # Average the gradients
        if total_count > 0:
            for p in self._network.parameters():
                if p.grad is not None:
                    p.grad.div_(total_count)
            return total_loss / total_count
        else:
            return 0.0

    ###Flatness Measurement
    def _with_rho_norm(self, d, rho, suffix="_norm"):
        """Return a copy of dict *d* with each scalar divided by *rho*."""
        return {k + suffix: v / rho for k, v in d.items()}

    # -----------------------------------------------------------------
    # Normalise a whole dictionary by the base loss 𝓛₀ (add key suffix)
    # -----------------------------------------------------------------
    def _with_L0_norm(self, d, L0, suffix="_norm"):
        """Return a copy of dict *d* with each scalar divided by *L0*."""
        eps = 1e-12                       # avoid div‑by‑zero if L0≈0
        return {k + suffix: v / (L0 + eps) for k, v in d.items()}
    # =========================================================
    # ---------------- helper: random delta -------------------
    # =========================================================
    def _sample_delta(self, shapes, rho, mode="linf"):
        """Return list of tensors whose concatenation has ‖·‖=rho."""
        flat = torch.randn(sum(np.prod(s) for s in shapes), device=self._device)
        if mode == "linf":
            flat = flat.sign() * rho
        else:                              # L2 sphere
            flat.mul_(rho / (flat.norm() + 1e-9))
        out, idx = [], 0
        for s in shapes:
            n = np.prod(s)
            out.append(flat[idx: idx+n].view(*s))
            idx += n
        return out


    # =========================================================
    # ---------------- 0‑order unified ------------------------
    # =========================================================
    def _zero_order_unified(self, loader, rho, n_dir, norm="linf"):
        plist  = [p for p in self._network.parameters() if p.requires_grad]
        shapes = [p.data.shape for p in plist]
        orig   = [p.data.clone() for p in plist]

        base_loss = self._evaluate_loss(loader)
        local, absl = [], []

        try:
            for _ in range(n_dir):
                delta_list = self._sample_delta(shapes, rho, norm)
                for p, d in zip(plist, delta_list): p.data.add_(d)
                loss_plus = self._evaluate_loss(loader)
                diff = loss_plus - base_loss
                local.append(diff); absl.append(abs(diff))
                # restore
                for p, o in zip(plist, orig): p.data.copy_(o)

            a = torch.as_tensor(local, device="cpu")
            b = torch.as_tensor(absl,  device="cpu")
            raw = {"mean": a.mean().item(), "std": a.std().item(), "max": a.max().item(),
                   "abs_mean": b.mean().item(), "abs_std": b.std().item(), "abs_max": b.max().item()}
            return {**raw, **self._with_L0_norm(raw, base_loss)}
        finally:
            for p, o in zip(plist, orig): p.data.copy_(o)

    # =========================================================
    # ------------- 0‑order SAM loss jump ---------------------
    # =========================================================
    def _zero_order_sam_loss(self, loader, rho):
        plist = [p for p in self._network.parameters() if p.requires_grad]
        saved = [p.data.clone() for p in plist]

        base = self._evaluate_loss(loader)
        _ = self._compute_old_tasks_loss_and_grad(loader)
        g    = [p.grad.clone() for p in plist]
        gnorm = torch.sqrt(sum(t.pow(2).sum() for t in g))
        if gnorm < 1e-9: return 0.0
        with torch.no_grad():
            for p, gi in zip(plist, g): p.data.add_(gi / gnorm, alpha=rho)
        loss = self._evaluate_loss(loader)
        for p, s in zip(plist, saved): p.data.copy_(s)
        return loss - base

    # =========================================================
    # ------------- 1‑st order unified (grad change) ----------
    # =========================================================
    def _first_order_unified(self, loader, rho, n_dir, norm="linf"):
        plist, shapes = [], []
        for p in self._network.parameters():
            if p.requires_grad: plist.append(p); shapes.append(p.data.shape)
        saved = [p.data.clone() for p in plist]

        base_loss = self._evaluate_loss(loader)

        # gradient at θ
        _ = self._compute_old_tasks_loss_and_grad(loader)
        g0 = [p.grad.clone() for p in plist]

        diffs = []
        try:
            for _ in range(n_dir):
                delta = self._sample_delta(shapes, rho, norm)
                for p, d in zip(plist, delta): p.data.add_(d)
                _ = self._compute_old_tasks_loss_and_grad(loader)
                g1 = [p.grad.clone() for p in plist]
                d  = torch.sqrt(sum((a-b).pow(2).sum() for a, b in zip(g1, g0)))
                diffs.append(d.item())
                for p, s in zip(plist, saved): p.data.copy_(s)

            t = torch.as_tensor(diffs, device="cpu")
            abs_t = t.abs()
            raw = {"mean": t.mean().item(), "std": t.std().item(), "max": t.max().item(),
                   "abs_mean": abs_t.mean().item(), "abs_std": abs_t.std().item(), "abs_max": abs_t.max().item()}
            return {**raw, **self._with_L0_norm(raw, base_loss)}
        finally:
            for p, s in zip(plist, saved): p.data.copy_(s)

    # =========================================================
    # ------------- 1‑st order SAM gradient change ------------
    # =========================================================
    def _first_order_sam_grad(self, loader, rho):
        plist = [p for p in self._network.parameters() if p.requires_grad]
        saved = [p.data.clone() for p in plist]

        _ = self._compute_old_tasks_loss_and_grad(loader)
        g0 = [p.grad.clone() for p in plist]
        gnorm = torch.sqrt(sum(t.pow(2).sum() for t in g0))
        if gnorm < 1e-9: return 0.0

        with torch.no_grad():
            for p, g in zip(plist, g0): p.data.add_(g / gnorm, alpha=rho)
        _ = self._compute_old_tasks_loss_and_grad(loader)
        g1 = [p.grad.clone() for p in plist]

        diff = torch.sqrt(sum((a-b).pow(2).sum() for a, b in zip(g1, g0))).item()
        for p, s in zip(plist, saved): p.data.copy_(s)
        return diff