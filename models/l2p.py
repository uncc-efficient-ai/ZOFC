import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import PromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8

import wandb

class Learner(BaseLearner):


    def __init__(self, args):
        super().__init__(args)
    
        self._network = PromptVitNet(args, True)

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

        # Freeze the parameters for ViT.
        if self.args["freeze"]:
            for p in self._network.original_backbone.parameters():
                p.requires_grad = False
        
            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in self._network.backbone.named_parameters():
                if n.startswith(tuple(self.args["freeze"])):
                    p.requires_grad = False
        
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

        wandb.init(project="l2p_mem", config=args)
        self.global_step = 0

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        if self.trainer == "ZO":  # <-- NEW
            self._zeroth_order_train(train_loader, test_loader)
            return

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
            
        if self._cur_task > 0:
            self._init_prompt(optimizer)

        if self._cur_task > 0 and self.args["reinit_optimizer"]:
            optimizer = self.get_optimizer()
            
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def get_optimizer(self):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                momentum=0.9, 
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )
            
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr, 
                weight_decay=self.weight_decay
            )

        return optimizer
    
    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_prompt(self, optimizer):
        args = self.args
        model = self._network.backbone
        task_id = self._cur_task

        # Transfer previous learned prompt params to the new prompt
        if args["prompt_pool"] and args["shared_prompt_pool"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

                with torch.no_grad():
                    model.prompt.prompt.grad.zero_()
                    model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                    optimizer.param_groups[0]['params'] = model.parameters()
                
        # Transfer previous learned prompt param keys to the new prompt
        if args["prompt_pool"] and args["shared_prompt_key"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

            with torch.no_grad():
                model.prompt.prompt_key.grad.zero_()
                model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                optimizer.param_groups[0]['params'] = model.parameters()

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):

            torch.cuda.reset_peak_memory_stats(device=self._device)

            self._network.backbone.train()
            self._network.original_backbone.eval()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
            
                output = self._network(inputs, task_id=self._cur_task, train=True)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')

                loss = F.cross_entropy(logits, targets.long())
                if self.args["pull_constraint"] and 'reduce_sim' in output:
                    loss = loss - self.args["pull_constraint_coeff"] * output['reduce_sim']


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()

            peak_mem_bytes = torch.cuda.max_memory_allocated(device=self._device)
            peak_mem_mb = peak_mem_bytes / (1024 ** 2)

            train_loss = losses / len(train_loader)
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = None
            if (epoch + 1) % self.args["tuned_epoch"] == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)

                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}, peak_memory_MB{:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                    peak_mem_mb,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, peak_memory_MB{:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    peak_mem_mb,
                )
            #prog_bar.set_description(info)
            logging.info(info)

        current_lr = optimizer.param_groups[0]['lr']

        wandb_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "lr": current_lr,
            "peak_memory_MB": peak_mem_mb,
        }
        # If we computed test_acc this epoch, add it
        if test_acc is not None:
            wandb_dict["test_acc"] = test_acc

        wandb.log(wandb_dict)
        logging.info(info)

    # ----------------------------------------------------------------------
    #                 PURE ZEROTH-ORDER TRAINING (SPSA / Q-SPSA)
    # ----------------------------------------------------------------------
    def _zeroth_order_train(self, train_loader, test_loader):
        device = self._device
        epochs = self.args["tuned_epoch"]
        q = self.q_spsa
        eps = self.eps
        max_norm = self.max_norm

        # (1) build optimiser first (we’ll need it for _init_prompt)
        trainables = [p for p in self._network.parameters() if p.requires_grad]
        opt_flag = self.args.get("optimizer", "zo_sgd").lower()
        if opt_flag == "zo_sgd":
            opt = optim.SGD(trainables, lr=self.init_lr, momentum=0.9,
                            weight_decay=self.weight_decay)
        elif opt_flag == "zo_adam":
            opt = optim.Adam(trainables, lr=self.init_lr,
                             weight_decay=self.weight_decay)
        else:
            raise ValueError("optimizer must be 'zo_sgd' or 'zo_adam' in ZO mode")

        # (0) prompt warm-start for new task
        if self._cur_task > 0:
            self._init_prompt(opt)

        sch = self.get_scheduler(opt)

        # helper forward ---------------------------------------------------
        def _forward_loss(x, y):
            out = self._network(x, task_id=self._cur_task, train=True)
            logits = out["logits"][:, :self._total_classes]
            if self._known_classes:
                logits[:, :self._known_classes] = float('-inf')
            loss = F.cross_entropy(logits, y)
            if self.args.get("pull_constraint", False) and "reduce_sim" in out:
                loss = loss - self.args["pull_constraint_coeff"] * out["reduce_sim"]
            return loss, logits

        # pre-allocate grad buffers once
        g_buf_template = [torch.zeros_like(p) for p in trainables]

        self._network.to(device)
        prog_bar = tqdm(range(epochs))

        for ep in prog_bar:
            torch.cuda.reset_peak_memory_stats(device)


            self._network.backbone.train()
            self._network.original_backbone.eval()

            ep_loss, correct, total = 0.0, 0, 0

            for bi, (_, x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)

                # zero grad buffers
                for g in g_buf_template: g.zero_()
                lp_sum = lm_sum = 0.0

                # Q-SPSA ----------------------------------------------------
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
                    for g, d in zip(g_buf_template, deltas):
                        g.add_(diff * d)

                    lp_sum += lp.item();
                    lm_sum += lm.item()


                # write grads & clip ---------------------------------------
                scale = 1.0 / (2 * eps * q)
                for p, g in zip(trainables, g_buf_template):
                    p.grad = g.mul_(scale)

                # (optional) zero classifier rows ⟶ only if fc exists
                if self._cur_task > 0 and hasattr(self._network, "fc"):
                    with torch.no_grad():
                        if self._network.fc.weight.grad is not None:
                            self._network.fc.weight.grad[:self._known_classes].zero_()
                        if getattr(self._network.fc, "bias", None) is not None:
                            self._network.fc.bias.grad[:self._known_classes].zero_()

                torch.nn.utils.clip_grad_norm_(trainables, max_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)


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

            peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            tr_loss = ep_loss / len(train_loader)
            tr_acc = 100. * correct / total

            wandb.log({"epoch": ep + 1,
                       "train/epoch_loss": tr_loss,
                       "train/epoch_acc": tr_acc,
                       "peak_memory_gb": peak_mem},
                      step=self.global_step)

            info = (
                f"[ZO] Ep {ep + 1}/{epochs} loss={tr_loss:.3f} "
                f"train_acc={tr_acc:.1f} "
                f"gpu_mem={peak_mem:.2f}GB")
            logging.info(info)

        logging.info(f"[ZO] Finished {epochs} epochs.")

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
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
                outputs = model(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)