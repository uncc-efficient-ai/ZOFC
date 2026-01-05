import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.optim import Optimizer
import math
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import CodaPromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

import wandb

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8

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
        Measure around the *first batch of the epoch*:
          - weights_b: CUDA model/buffer tensors
          - grads_b:   .grad tensors
          - opt_b:     optimizer state tensors (momenta, Adam m/v, etc.)
          - optimizer = grads_b + opt_b
          - activations = max(forward-only caches, residual at global peak)
        """
        # convert
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

        # reserved + allocator overhead (to reconcile with nvidia-smi)
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
    
        self._network = CodaPromptVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args

        self.trainer = args.get("trainer", "FO").upper()
        if self.trainer != "FO":
            self.eps = args["eps"]
            self.q_spsa = args["q_spsa"]
            self.max_norm = args.get("max_norm", 1.0)

        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.fc.parameters() if p.requires_grad) + sum(p.numel() for p in self._network.prompt.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} fc and prompt training parameters.')

        wandb.init(
            project="coda_prompt_mem",
            config=self.args
        )
        self.global_step = 0

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1

        if self._cur_task > 0:
            try:
                if self._network.module.prompt is not None:
                    self._network.module.prompt.process_task_count()
            except:
                if self._network.prompt is not None:
                    self._network.prompt.process_task_count()

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        # ---- ZEROTH-ORDER path ---------------------------------
        if self.trainer == "ZO":
            self.data_weighting()
            self._zeroth_order_train(train_loader, test_loader)
            return
        # --------------------------------------------------------

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        self.data_weighting()
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def data_weighting(self):
        self.dw_k = torch.tensor(np.ones(self._total_classes + 1, dtype=np.float32))
        self.dw_k = self.dw_k.to(self._device)

    def get_optimizer(self):
        if len(self._multiple_gpus) > 1:
            params = list(self._network.module.prompt.parameters()) + list(self._network.module.fc.parameters())
        else:
            params = list(self._network.prompt.parameters()) + list(self._network.fc.parameters())
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(params, momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(params, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = CosineSchedule(optimizer, K=self.args["tuned_epoch"])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            #torch.cuda.reset_peak_memory_stats(device=self._device)
            # Reset allocator stats & sync
            self._reset_peaks()
            optimizer.zero_grad(set_to_none=True)
            self._cuda_sync()

            # Count weights on device once at start-of-epoch
            weights_b = self._bytes_model_weights(self._network)
            measured_this_epoch = False
            self._network.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                # logits
                logits, prompt_loss = self._network(inputs, train=True)
                logits = logits[:, :self._total_classes]

                logits[:, :self._known_classes] = float('-inf')
                dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
                loss_supervised = (F.cross_entropy(logits, targets.long()) * dw_cls).mean()

                # ce loss
                loss = loss_supervised + prompt_loss.sum()

                # forward-only peak on first batch
                if not measured_this_epoch and i == 0:
                    self._cuda_sync()
                    fwd_peak_b = torch.cuda.max_memory_allocated(self._device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                '''
                # global peak + buckets on first batch
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
                '''
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            peak_bytes = torch.cuda.max_memory_allocated(device=self._device)
            peak_gb = peak_bytes / (1024 ** 3)

            if (epoch + 1) % self.args['tuned_epoch'] == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            wandb.log({
                    "epoch": epoch + 1,
                    "train/acc": train_acc,
                    "peak_memory_gb": peak_gb,
            }, step=self.global_step)
            info = info + f", PeakMem {peak_gb:.2f}GB"
            logging.info(info)

        logging.info(info)

    # ZO SPSA TRAINING FOR CoDA-Prompt
    def _zeroth_order_train(self, train_loader, test_loader):
        device = self._device
        epochs = self.args["tuned_epoch"]
        q = self.q_spsa
        eps = self.eps
        max_norm = self.max_norm

        # build optimiser over prompt + fc (same as get_optimizer)
        trainables = [p for p in self._network.parameters() if p.requires_grad]
        flag = self.args.get("optimizer", "zo_sgd").lower()
        if flag == "zo_sgd":
            opt = optim.SGD(trainables, lr=self.init_lr, momentum=0.9,
                            weight_decay=self.weight_decay)
        else:
            opt = optim.Adam(trainables, lr=self.init_lr,
                             weight_decay=self.weight_decay)
        sch = self.get_scheduler(opt)

        # helper forward
        def _forward_loss(x, y):
            logits, prompt_loss = self._network(x, train=True)  # CoDA net returns both
            logits = logits[:, :self._total_classes]
            if self._known_classes:
                logits[:, :self._known_classes] = float('-inf')

            dw = self.dw_k[-1 * torch.ones_like(y).long()]
            ce = (F.cross_entropy(logits, y, reduction='none') * dw).mean()
            total = ce + prompt_loss.sum()
            return total, logits

        # single grad buffer reused every batch
        g_buf = [torch.zeros_like(p) for p in trainables]

        self._network.to(device)
        prog = tqdm(range(epochs))

        for ep in prog:
            torch.cuda.reset_peak_memory_stats(device)

            self._network.train()

            ep_loss, correct, total = 0.0, 0, 0

            for _, (_, x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                for g in g_buf: g.zero_()  # reset accumulators
                lp_sum = lm_sum = 0.0

                # Q-SPSA inner loop
                for _ in range(q):
                    deltas = [torch.randint_like(p, 0, 2).mul_(2).sub_(1)
                              for p in trainables]

                    for p, d in zip(trainables, deltas): p.data.add_(eps * d)
                    with torch.no_grad():
                        lp, _ = _forward_loss(x, y)

                    for p, d in zip(trainables, deltas): p.data.add_(-2 * eps * d)
                    with torch.no_grad():
                        lm, _ = _forward_loss(x, y)

                    for p, d in zip(trainables, deltas): p.data.add_(eps * d)

                    diff = (lp - lm)
                    for g, d in zip(g_buf, deltas):
                        g.add_(diff * d)

                    lp_sum += lp.item()
                    lm_sum += lm.item()
                #



                scale = 1.0 / (2 * eps * q)
                for p, g in zip(trainables, g_buf):
                    p.grad = g.mul_(scale)

                torch.nn.utils.clip_grad_norm_(trainables, max_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)


                # batch stats
                with torch.no_grad():
                    loss_b, logits_b = _forward_loss(x, y)
                ep_loss += loss_b.item()
                correct += logits_b.argmax(1).eq(y).sum().item()
                total += y.size(0)

                self.global_step += 1
                wandb.log({"train/batch_loss": loss_b.item(),
                           "train/loss_plus": lp_sum / q,
                           "train/loss_minus": lm_sum / q},
                          step=self.global_step)

            if sch: sch.step()

            peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            tr_loss = ep_loss / len(train_loader)
            tr_acc = 100. * correct / total

            wandb.log({"epoch": ep + 1,
                       "train/epoch_loss": tr_loss,
                       "train/epoch_acc": tr_acc,
                       "peak_memory_gb": peak},
                      step=self.global_step)

            info = (
                f"[ZO] Ep {ep + 1}/{epochs} loss={tr_loss:.3f} "
                f"train_acc={tr_acc:.1f}"
                f"gpu_mem={peak:.2f}GB")
            logging.info(info)

        logging.info(f"[ZO] Finished {epochs} epochs.")

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)[:, :self._total_classes]
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
            with torch.no_grad():
                outputs = model(inputs)[:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineSchedule(_LRScheduler):

    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (self.K-1)))

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]