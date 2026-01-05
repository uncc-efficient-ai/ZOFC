import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,MultiBranchCosineIncrementalNet,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

import wandb

# tune the model at first session with adapter, and then conduct simplecil.
num_workers = 8

class Learner(BaseLearner):

    # Memory helpers
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
        Log memory buckets (first batch of the epoch):
          - Weights (parameters+buffers)
          - Optimizer = grads + optimizer states
          - Activations = max(forward-only caches, residual at global peak)
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

        # Console
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
        if 'adapter' not in args["backbone_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')

        if 'resnet' in args['backbone_type']:
            self._network = SimpleCosineIncrementalNet(args, True)
            self. batch_size=128
            self.init_lr=args["init_lr"] if args["init_lr"] is not None else  0.01
        else:
            self._network = SimpleVitNet(args, True)
            self. batch_size= args["batch_size"]
            self. init_lr=args["init_lr"]
        
        self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args=args


        # Zeroth-order config (SPSA)
        self.eps = args.get("eps", 0.001)         # perturbation scale
        self.q_spsa = args.get("q_spsa", 4)      # number of repeated perturbations


        wandb.init(
            project="aper_adapter_mem",
            config=self.args
        )
        self.global_step = 0

    def after_task(self):
        self._known_classes = self._total_classes
    
    def replace_fc(self,trainloader, model, args):
        # replace fc.weight with the embedding average of train data
        model = model.eval()
        embedding_list = []
        label_list = []
        # data_list=[]
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.to(self._device)
                label=label.to(self._device)
                embedding = model(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list=np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            proto=embedding.mean(0)
            self._network.fc.weight.data[class_index]=proto
        return model

    

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        
        self._network.to(self._device)
        
        if self._cur_task == 0:
            # show total parameters and trainable parameters
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            if total_params != total_trainable_params:
                for name, param in self._network.named_parameters():
                    if param.requires_grad:
                        print(name, param.numel())


            if self.args.get('trainer', '') == 'hybrid_spsa':

                # Construct separate param groups:
                #   - (1) adapter-only
                #   - (2) classifier-only
                adapter_params, fc_params = self._split_adapter_and_classifier_params()

                # Build separate optimizers & schedulers
                # 1) Classifier (FO) => same epochs as original method: self.args['tuned_epoch']
                fc_optimizer = self._build_optimizer(fc_params, kind="classifier")
                fc_scheduler = self._build_scheduler(fc_optimizer, self.args['tuned_epoch'])

                # 2) Adapter (ZO) => train for self.args['adapter_epoch'] if you want it longer
                #    If "adapter_epoch" not given, fall back to e.g. 2 * tuned_epoch
                adapter_epochs = self.args.get("adapter_epoch", 2 * self.args['tuned_epoch'])
                adapter_optimizer = self._build_optimizer(adapter_params, kind="adapter")
                adapter_scheduler = self._build_scheduler(adapter_optimizer, adapter_epochs)

                self._train_hybrid_spsa(
                    train_loader,
                    test_loader,
                    adapter_optimizer,
                    adapter_scheduler,
                    fc_optimizer,
                    fc_scheduler,
                    adapter_epochs
                )
            elif self.args['optimizer'] in ['zo_sgd', 'zo_adam']:
                # Build an optimizer just like normal
                if self.args['optimizer']=='zo_sgd':
                    optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
                elif self.args['optimizer']=='zo_adam':
                    optimizer=optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
                scheduler=None
                self._zeroth_order_train(train_loader, test_loader, optimizer, scheduler)
            else:

                if self.args['optimizer']=='sgd':
                    optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
                elif self.args['optimizer']=='adam':
                    optimizer=optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
                scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)

                self._init_train(train_loader, test_loader, optimizer, scheduler)
            self.construct_dual_branch_network()
        else:
            pass
        self.replace_fc(train_loader_for_protonet, self._network, None)
            

    def construct_dual_branch_network(self):
        network = MultiBranchCosineIncrementalNet(self.args, True)
        network.construct_dual_branch_network(self._network)
        self._network=network.to(self._device)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):

            #torch.cuda.reset_peak_memory_stats(self._device)
            # ── NEW: epoch baseline & peaks reset
            self._reset_peaks()
            optimizer.zero_grad(set_to_none=True)
            self._cuda_sync()
            weights_b = self._bytes_model_weights(self._network)
            measured_this_epoch = False
            # ─────────────────────────────────────

            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)

                # forward-only peak on the first batch
                if not measured_this_epoch and i == 0:
                    self._cuda_sync()
                    fwd_peak_b = torch.cuda.max_memory_allocated(self._device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                '''
                # LOG full buckets after first update
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
                #
                '''

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()

            # 2) Check peak GPU memory usage (in MB) after finishing the training loop
            peak_mem = torch.cuda.max_memory_allocated(self._device) / (1024 ** 2)

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(self._network, test_loader)

            wandb.log({
                "epoch": epoch,
                "train/loss": losses / len(train_loader),
                "train/acc": train_acc,
                "test/acc": test_acc,
                "learning_rate": scheduler.get_last_lr()[0],
                "peak_mem_MB_fo": peak_mem
            }, step=self.global_step)

            self.global_step += 1

            info = (
                f"Task {self._cur_task}, Epoch {epoch + 1}/{self.args['tuned_epoch']} => "
                f"Loss {losses / len(train_loader):.3f}, Train_acc {train_acc:.2f}, "
                f"Test_acc {test_acc:.2f}, PeakMem {peak_mem:.2f}MB"
            )
            #prog_bar.set_description(info)
            logging.info(info)

        logging.info(info)

    def _zeroth_order_train(self, train_loader, test_loader, optimizer, scheduler):
        epochs = self.args['tuned_epoch']
        q_spsa = self.q_spsa
        eps = self.eps

        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:

            ########################################################################
            # Reset peak memory stats at the start of each epoch
            #torch.cuda.reset_peak_memory_stats(self._device)
            # reset peaks & make a clean baseline
            self._reset_peaks()
            optimizer.zero_grad(set_to_none=True)
            self._cuda_sync()
            weights_b = self._bytes_model_weights(self._network)
            measured_this_epoch = False
            #
            ########################################################################

            self._network.train()
            epoch_loss = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                # 1) Collect trainable parameters
                param_list = [p for p in self._network.parameters() if p.requires_grad]
                # 2) We'll accumulate approximate gradient in 'param_grads'
                param_grads = [torch.zeros_like(p.data) for p in param_list]

                # 3) Q-SPSA loop
                for _ in range(q_spsa):
                    # Sample Rademacher deltas
                    deltas = []
                    for p in param_list:
                        delta = torch.randint(low=0, high=2, size=p.shape, device=p.device, dtype=p.dtype)
                        delta = delta * 2 - 1  # convert {0,1} => {-1,+1}
                        deltas.append(delta)

                    # Evaluate loss_plus at (theta + eps*delta)
                    for p, d in zip(param_list, deltas):
                        p.data.add_(eps * d)
                    with torch.no_grad():
                        logits_plus = self._network(inputs)["logits"]
                        loss_plus = F.cross_entropy(logits_plus, targets)

                    # Evaluate loss_minus at (theta - eps*delta)
                    for p, d in zip(param_list, deltas):
                        p.data.sub_(2.0 * eps * d)  # net shift from +eps to -eps
                    with torch.no_grad():
                        logits_minus = self._network(inputs)["logits"]
                        loss_minus = F.cross_entropy(logits_minus, targets)

                    # Restore original params => +eps*d
                    for p, d in zip(param_list, deltas):
                        p.data.add_(eps * d)

                    # Accumulate finite-diff gradient
                    diff = (loss_plus - loss_minus)
                    for idx, d in enumerate(deltas):
                        param_grads[idx].add_(diff * d)

                # forward-only peak on the first batch (after SPSA fwd passes)
                if not measured_this_epoch and i == 0:
                    self._cuda_sync()
                    fwd_peak_b = torch.cuda.max_memory_allocated(self._device)

                # 4) Scale the gradient => 1/(2*eps*q_spsa)
                scale_factor = 1.0 / (2.0 * eps * q_spsa)
                for idx, p in enumerate(param_list):
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    p.grad.detach_()
                    p.grad.copy_(param_grads[idx] * scale_factor)

                # 5) Optional grad clip
                torch.nn.utils.clip_grad_norm_(param_list, max_norm=1.0)

                # 6) Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                # ── NEW: global peak + buckets on first batch
                if not measured_this_epoch and i == 0:
                    self._cuda_sync()
                    peak_b = torch.cuda.max_memory_allocated(self._device)
                    grads_b = self._bytes_grads(self._network)
                    opt_b = self._bytes_optimizer_states(optimizer)
                    self._log_mem_buckets(
                        tag=f"mem/ZO_epoch{epoch + 1}",
                        weights_b=weights_b, grads_b=grads_b, opt_b=opt_b,
                        fwd_peak_b=fwd_peak_b, peak_b=peak_b
                    )
                    measured_this_epoch = True
                # ─────────────────────────────────────

                # Evaluate on this batch after the update
                with torch.no_grad():
                    logits_current = self._network(inputs)["logits"]
                    loss_batch = F.cross_entropy(logits_current, targets)
                    epoch_loss += loss_batch.item()

                    # Batch accuracy
                    _, preds = torch.max(logits_current, dim=1)
                    correct += preds.eq(targets).sum().item()
                    total += len(targets)

            # End of epoch
            if scheduler:
                scheduler.step()

            ########################################################################
            # Query peak GPU usage in MB after epoch finishes
            peak_mem = torch.cuda.max_memory_allocated(self._device) / (1024 ** 2)
            ########################################################################

            train_loss = epoch_loss / len(train_loader)
            train_acc = (100.0 * correct / total) if total > 0 else 0.0

            # Evaluate on test set
            test_acc = self._compute_accuracy(self._network, test_loader)

            # Logging
            wandb.log({
                "epoch": epoch + 1,
                "train_loss_zo": train_loss,
                "train_acc_zo": train_acc,
                "test_acc_zo": test_acc,
                "peak_mem_MB_zo": peak_mem
            }, step=self.global_step)
            self.global_step += 1

            info = (f"[ZO-APER] Epoch {epoch + 1}/{epochs} => "
                    f"Train_Loss={train_loss:.3f}, Train_Acc={train_acc:.2f}, "
                    f"Test_Acc={test_acc:.2f}, PeakMem={peak_mem:.2f}MB")
            prog_bar.set_description(info)
            logging.info(info)

        logging.info(f"[ZO-APER] Finished SPSA training for {epochs} epochs.")


    def _train_hybrid_spsa(
            self,
            train_loader,
            test_loader,
            adapter_optimizer,
            adapter_scheduler,
            fc_optimizer,
            fc_scheduler,
            adapter_epochs
    ):
        """
        Hybrid approach for the first task:
          - We run a total of adapter_epochs.
          - For the first 'tuned_epoch' epochs, we do FO on classifier + ZO on adapter simultaneously.
          - After 'tuned_epoch' epochs, we freeze classifier and continue ZO on adapter only
            for the remaining (adapter_epochs - tuned_epoch) epochs.
        """

        tuned_epoch = self.args['tuned_epoch']  # e.g. 20

        logging.info(f"[Hybrid] Unified loop => total {adapter_epochs} epochs. "
                     f"Classifier trains FO for first {tuned_epoch} epochs, then frozen.")

        prog_bar = tqdm(range(self.args['adapter_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            # -------------------------------------------------------------
            # (A) Decide whether classifier is trainable this epoch
            # -------------------------------------------------------------
            #if epoch < tuned_epoch:
                # Unfreeze classifier (FO)
            #    self._freeze_or_unfreeze_fc(True)
            #else:
                # Freeze classifier after tuned_epoch
            #    self._freeze_or_unfreeze_fc(False)

            # -------------------------------------------------------------
            # (B) Adapter is always ZO across the entire range (0..adapter_epochs-1),
            #     or if you want to skip adapter updates before tuned_epoch, adjust logic
            # -------------------------------------------------------------
            self._freeze_or_unfreeze_adapter(True)

            # Collect adapter params for the SPSA logic
            adapter_params = [p for p in self._network.parameters() if p.requires_grad]

            # For each mini-batch
            for _, inputs, targets in self._iter_batches(train_loader):
                # 1) If classifier is still trainable => standard FO step
                if epoch < tuned_epoch:
                    fc_optimizer.zero_grad()
                    logits_fo = self._network(inputs)["logits"]
                    loss_fo = F.cross_entropy(logits_fo, targets)
                    loss_fo.backward()  # normal backprop for classifier
                    fc_optimizer.step()

                # 2) Now do a single ZO (SPSA) step on the adapter
                #    We'll accumulate approximate gradients in 'param_grads'
                adapter_optimizer.zero_grad()
                param_grads = [torch.zeros_like(p.data) for p in adapter_params]

                loss_plus_total = 0.0
                loss_minus_total = 0.0

                for _ in range(self.q_spsa):
                    # Sample Rademacher deltas
                    deltas = []
                    for p in adapter_params:
                        d = torch.randint(0, 2, p.shape, device=p.device, dtype=p.dtype)
                        d = d * 2 - 1  # {0,1} => {-1,+1}
                        deltas.append(d)

                    # SHIFT +eps
                    for p, d in zip(adapter_params, deltas):
                        p.data.add_(self.eps * d)

                    with torch.no_grad():
                        logits_plus = self._network(inputs)["logits"]
                        loss_plus = F.cross_entropy(logits_plus, targets)
                    loss_plus_total += loss_plus.item()

                    # SHIFT -eps => net shift of -2.0 * eps
                    for p, d in zip(adapter_params, deltas):
                        p.data.add_(-2.0 * self.eps * d)

                    with torch.no_grad():
                        logits_minus = self._network(inputs)["logits"]
                        loss_minus = F.cross_entropy(logits_minus, targets)
                    loss_minus_total += loss_minus.item()

                    # Restore original params (add +eps*d)
                    for p, d in zip(adapter_params, deltas):
                        p.data.add_(self.eps * d)

                    # Approx grad: diff * deltas
                    diff = (loss_plus - loss_minus)
                    for idx, p in enumerate(adapter_params):
                        param_grads[idx].add_(diff * deltas[idx])

                # Scale the accumulated gradients: 1/(2*eps*q_spsa)
                scale_factor = 1.0 / (2.0 * self.eps * self.q_spsa)
                for idx, p in enumerate(adapter_params):
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    p.grad.detach_()
                    p.grad.copy_(param_grads[idx] * scale_factor)

                # Clip adapter grad if desired
                torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=5.0)

                # Apply the adapter update
                adapter_optimizer.step()

                # Evaluate updated model on the same batch
                with torch.no_grad():
                    logits_updated = self._network(inputs)["logits"]
                    loss_batch = F.cross_entropy(logits_updated, targets)

                losses += loss_batch.item()
                _, preds = torch.max(logits_updated, dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)

            # (C) Step LR schedulers for classifier & adapter
            # Classifier scheduler only up to tuned_epoch
            if epoch < tuned_epoch and fc_scheduler is not None:
                fc_scheduler.step()
            if adapter_scheduler is not None:
                adapter_scheduler.step()

            # (D) Compute train/test accuracy after this epoch
            train_acc = np.around(correct * 100 / total, decimals=2) if total else 0.0
            test_acc = self._compute_accuracy(self._network, test_loader)

            # W&B logging
            self.global_step += 1
            wandb.log({
                "epoch": epoch + 1,
                "adapter/loss": losses / len(train_loader),
                "adapter/train_acc": train_acc,
                "adapter/test_acc": test_acc
            }, step=self.global_step)

            info = (f"[Hybrid][Epoch {epoch + 1}/{adapter_epochs}] => "
                    f"Loss {losses / len(train_loader):.3f}, "
                    f"Train_acc {train_acc:.2f}, Test_acc {test_acc:.2f}")
            logging.info(info)

        logging.info(f"[Hybrid] Finished unified loop of {adapter_epochs} epochs.")
        # Unfreeze everything if needed
        self._freeze_or_unfreeze_fc(True)
        self._freeze_or_unfreeze_adapter(True)

    def _iter_batches(self, loader):
        """
        Utility generator that yields (index, inputs, targets)
        with data on self._device for convenience
        """
        for i, (_, x, y) in enumerate(loader):
            yield i, x.to(self._device), y.to(self._device)

    # --------------------------------------------------------------------
    # HELPER: Separate adapter vs. classifier parameters
    # --------------------------------------------------------------------
    def _split_adapter_and_classifier_params(self):
        """
        Returns two lists: adapter_params, fc_params
        """
        adapter_params = []
        fc_params = []
        for name, param in self._network.named_parameters():
            if "adaptmlp" in name:
                adapter_params.append(param)
            elif "fc" in name:
                fc_params.append(param)
            # else: it's part of the backbone (which might or might not be trainable,
            # but typically is mostly frozen if using an adapter approach)
        return adapter_params, fc_params

    # --------------------------------------------------------------------
    # HELPER: Build an optimizer for either classifier or adapter
    # --------------------------------------------------------------------
    def _build_optimizer(self, params, kind="classifier"):
        """
        kind can be 'classifier' or 'adapter'—you can vary the LR or optimizer type
        if you want. Here we do something simple, but you can expand.
        """
        # If the user wants to set separate LRs:
        if kind == "classifier":
            lr = self.args.get("fc_lr", self.init_lr)
        else:  # adapter
            lr = self.args.get("adapter_lr", self.init_lr)

        # Reuse self.args['optimizer'] to pick type
        opt_type = self.args.get("optimizer", "sgd")

        if opt_type == ["sgd", "zo_sgd"]:
            optimizer = optim.SGD(params,
                                  lr=lr,
                                  momentum=0.9,
                                  weight_decay=self.weight_decay)
        elif opt_type in ["adam", "zo_adam"]:
            optimizer = optim.Adam(params, lr=lr, weight_decay=self.weight_decay)
        elif opt_type == "adamw":
            optimizer = optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        else:
            # fallback
            optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=self.weight_decay)
        return optimizer

    # --------------------------------------------------------------------
    # HELPER: Build a scheduler
    # --------------------------------------------------------------------
    def _build_scheduler(self, optimizer, max_epochs):
        scheduler_type = self.args.get("scheduler", "cosine")
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=max_epochs, eta_min=self.min_lr
            )
        elif scheduler_type == 'steplr':
            return optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args.get("init_milestones", [30, 60]),
                gamma=self.args.get("init_lr_decay", 0.1)
            )
        elif scheduler_type == 'constant':
            return None
        else:
            return None

    # --------------------------------------------------------------------
    # HELPER: Freeze/unfreeze
    # --------------------------------------------------------------------
    def _freeze_or_unfreeze_adapter(self, unfreeze: bool):
        """
        If `unfreeze=True`, adapter params get requires_grad_(True).
        If `unfreeze=False`, adapter params get requires_grad_(False).
        """
        for name, param in self._network.named_parameters():
            if "adapter" in name or "lora" in name or "prefix" in name:
                param.requires_grad = unfreeze

    def _freeze_or_unfreeze_fc(self, unfreeze: bool):
        """
        If `unfreeze=True`, classifier (FC) params get requires_grad_(True).
        If `unfreeze=False`, classifier (FC) params get requires_grad_(False).
        """
        for name, param in self._network.named_parameters():
            if "fc" in name:
                param.requires_grad = unfreeze
