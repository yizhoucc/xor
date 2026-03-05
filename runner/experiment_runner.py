from __future__ import division, print_function
import os
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

from model import *
from dataset import *
from utils.logger import get_logger
from utils.train_helper import snapshot, load_model, EarlyStopper
from utils.corpus import Corpus

logger = get_logger('exp_logger')

__all__ = ['ExperimentRunner']


class ExperimentRunner:
    """Unified experiment runner for all model types (baseline, 1-arg, 2-arg)
    across all architectures (MLP, CNN, RNN).

    Handles the full pipeline: pretrain → phase1 → phase2 → test
    with automatic skipping for baseline models.
    """

    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        self.dataset_conf = config.dataset
        self.model_conf = config.model
        self.pretrain_conf = config.pretrain
        self.train_conf = config.train
        self.train_phase2_conf = config.get('train_phase2', {})
        self.test_conf = config.test
        self.use_gpu = config.use_gpu
        self.gpus = config.gpus
        self.save_dir = config.save_dir

        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.is_rnn = (self.dataset_conf.name == 'ptb')
        self.has_inner_net = self.model_conf.name not in ('BaselineMLP', 'BaselineCNN', 'BaselineRNN')

        # For RNN tasks, load corpus
        if self.is_rnn:
            self.corpus = Corpus(self.dataset_conf.data_path + '/ptb')
            self.ntokens = len(self.corpus.dictionary)

    # ==================== Pretrain (Session I) ==================== #

    def pretrain(self):
        """Pretrain InnerNet on random smoothed functions.
        Skipped for baseline models.
        """
        if not self.has_inner_net:
            print("Baseline model — skipping pretrain.")
            self._mark_stage('PRETRAIN_DONE')
            return

        num_cell_types = getattr(self.model_conf, 'num_cell_types', 1)
        for cell_type in range(num_cell_types):
            self._pretrain_single(cell_type)

        self._mark_stage('PRETRAIN_DONE')

    def _pretrain_single(self, cell_type):
        """Pretrain a single InnerNet instance."""
        self.config.seed = int(str(self.seed) + str(cell_type))
        train_dataset = InnerNetData(self.config, split='train')
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.pretrain_conf.batch_size,
            shuffle=self.pretrain_conf.shuffle,
            num_workers=self.pretrain_conf.num_workers,
            collate_fn=train_dataset.collate_fn,
            drop_last=False)

        model = InnerNet(self.config)
        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.pretrain_conf.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=self.pretrain_conf.lr,
                                   weight_decay=self.pretrain_conf.wd)
        elif self.pretrain_conf.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=self.pretrain_conf.lr,
                                   momentum=self.pretrain_conf.momentum,
                                   weight_decay=self.pretrain_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        best_train_loss = np.inf
        for epoch in range(self.pretrain_conf.max_epoch):
            train_loss = []
            model.train()
            for xy, targets in train_loader:
                optimizer.zero_grad(set_to_none=True)
                if self.use_gpu:
                    xy, targets = xy.cuda(), targets.cuda()
                _, loss = model(xy, targets)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            train_loss = np.mean(train_loss)
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                logger.info(f"Pretrain Loss @ epoch {epoch + 1:04d} = {best_train_loss:.6f}")
                snapshot(
                    model.module.inner_net if self.use_gpu else model.inner_net,
                    optimizer, self.config, epoch + 1,
                    tag=f'best_pretrained{cell_type}')
            elif (epoch + 1) % 100 == 0:
                logger.info(f"Pretrain epoch {epoch + 1:04d} (no improvement, best = {best_train_loss:.6f})")

    # ==================== Phase 1 (Session II) ==================== #

    def train_phase1(self):
        """Joint training (or baseline regular training).
        For 2-arg/1-arg: load pretrained InnerNet, then jointly train.
        For baseline: standard training from scratch.
        """
        self.config.seed = self.seed

        if self.is_rnn:
            best_val_loss = self._train_phase1_rnn()
        else:
            best_val_loss = self._train_phase1_classification()

        self._mark_stage('PHASE1_DONE')
        return best_val_loss

    def _train_phase1_classification(self):
        """Phase 1 training for classification tasks (MNIST, CIFAR-10)."""
        train_loader, val_loader = self._get_classification_loaders()

        # Create model
        model = eval(self.model_conf.name)(self.config)

        # Load pretrained InnerNet if applicable
        if self.has_inner_net:
            num_cell_types = getattr(self.model_conf, 'num_cell_types', 1)
            if hasattr(model, 'inner_net') and isinstance(model.inner_net, nn.ModuleList):
                for i in range(num_cell_types):
                    load_model(model.inner_net[i],
                               self.save_dir + self.pretrain_conf.best_model[i])
            elif hasattr(model, 'inner_net') and isinstance(model.inner_net, nn.Sequential):
                # XorNeuronMLP/XorNeuronConv with single shared InnerNet
                load_model(model.inner_net,
                           self.save_dir + self.pretrain_conf.best_model[0])

        model = model.to(self.device)
        if self.use_gpu and len(self.gpus) > 1:
            model = nn.DataParallel(model, device_ids=self.gpus)

        optimizer = self._make_optimizer(model)
        early_stop = EarlyStopper([0.0], win_size=self.train_conf.get('early_stop_window', 20), is_decrease=False)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.train_conf.lr_decay_steps,
            gamma=self.train_conf.lr_decay)

        iter_count = 0
        results = defaultdict(list)
        best_val_loss = float('inf')
        best_val_acc = 0.0

        for epoch in range(self.train_conf.max_epoch):
            # Validation
            if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
                model.eval()
                val_loss_sum = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs = imgs.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        out, loss, _ = model(imgs, labels)
                        val_loss_sum += loss.item()
                        _, pred = torch.max(out, 1)
                        total += labels.size(0)
                        correct += (pred == labels).sum().item()

                avg_val_loss = val_loss_sum / len(val_loader)
                acc = correct / total
                results['val_loss'].append(avg_val_loss)
                results['val_acc'].append(acc)

                logger.info(f"Epoch {epoch+1} | Val Loss = {avg_val_loss:.6f} | Val Acc = {acc:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = acc
                    model_to_save = model.module if hasattr(model, 'module') else model
                    if self.has_inner_net:
                        snapshot(model_to_save.inner_net, optimizer, self.config,
                                 epoch + 1, tag='best_phase1')
                    else:
                        snapshot(model_to_save, optimizer, self.config,
                                 epoch + 1, tag='best_phase1')

                if early_stop.tick([avg_val_loss]):
                    logger.info("Early stopping triggered.")
                    break

            # Training
            model.train()
            lr_scheduler.step()

            for imgs, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                _, loss, _ = model(imgs, labels)
                loss.backward()
                optimizer.step()

                results['train_loss'].append(loss.item())
                results['train_step'].append(iter_count)

                if (iter_count + 1) % self.train_conf.display_iter == 0:
                    logger.info(f"Train Loss @ epoch {epoch+1:04d} iter {iter_count+1:08d} = {loss.item():.6f}")

                iter_count += 1

        results['best_val_loss'].append(best_val_loss)
        results['best_val_acc'].append(best_val_acc)
        pickle.dump(results, open(os.path.join(self.save_dir, 'train_stats_phase1.p'), 'wb'))
        logger.info(f"Phase1 Best Val Loss = {best_val_loss:.6f} | Best Val Acc = {best_val_acc:.4f}")

        return best_val_loss

    def _train_phase1_rnn(self):
        """Phase 1 training for RNN language modeling (PTB)."""
        train_data = self._batchify(self.corpus.train, self.train_conf.batch_size)
        val_data = self._batchify(self.corpus.valid, self.train_conf.batch_size)

        # Create model
        model = eval(self.model_conf.name)(self.config, self.ntokens)

        # Load pretrained InnerNet
        if self.has_inner_net:
            num_cell_types = getattr(self.model_conf, 'num_cell_types', 1)
            for i in range(num_cell_types):
                load_model(model.inner_net[i],
                           self.save_dir + self.pretrain_conf.best_model[i])

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        optimizer = self._make_optimizer(model)
        early_stop = EarlyStopper([0.0], win_size=self.train_conf.get('early_stop_window', 20), is_decrease=False)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.train_conf.lr_decay_steps,
            gamma=self.train_conf.lr_decay)

        iter_count = 0
        results = defaultdict(list)
        best_val_loss = np.inf

        for epoch in range(self.train_conf.max_epoch):
            # Validation
            if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
                model.eval()
                val_loss = []
                hidden = None
                for i in range(0, val_data.size(0) - 1, self.train_conf.bptt):
                    data, targets = self._get_batch(val_data, i)
                    if self.use_gpu:
                        data, targets = data.cuda(), targets.cuda()
                    with torch.no_grad():
                        _, hidden, loss, _ = model(data, targets, mask=None, hx=hidden)
                    hidden = self._repackage_hidden(hidden)
                    val_loss.append(loss.item())

                avg_val_loss = np.mean(val_loss)
                results['val_loss'].append(avg_val_loss)
                logger.info(f"Epoch {epoch+1} | Val Loss = {avg_val_loss:.6f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model_to_save = model.module if hasattr(model, 'module') else model
                    if self.has_inner_net:
                        snapshot(model_to_save.inner_net, optimizer, self.config,
                                 epoch + 1, tag='best_phase1')
                    else:
                        snapshot(model_to_save, optimizer, self.config,
                                 epoch + 1, tag='best_phase1')

                if early_stop.tick([avg_val_loss]):
                    logger.info("Early stopping triggered.")
                    break

            # Training
            model.train()
            lr_scheduler.step()
            hidden = None
            mask = [None] * len(self.model_conf.out_hidden_dim)
            for i in range(len(self.model_conf.out_hidden_dim)):
                mask[i] = (torch.bernoulli(
                    torch.Tensor(self.model_conf.out_hidden_dim[i]).fill_(
                        1 - self.model_conf.dropout))
                    / (1 - self.model_conf.dropout))

            for batch, i in enumerate(range(0, train_data.size(0) - 1, self.train_conf.bptt)):
                optimizer.zero_grad(set_to_none=True)
                data, targets = self._get_batch(train_data, i)
                if self.use_gpu:
                    data, targets = data.cuda(), targets.cuda()
                _, hidden, loss, _ = model(data, targets, mask=mask, hx=hidden)
                hidden = self._repackage_hidden(hidden)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.train_conf.clip)
                optimizer.step()

                results['train_loss'].append(loss.item())
                results['train_step'].append(iter_count)

                if (iter_count + 1) % self.train_conf.display_iter == 0:
                    logger.info(f"Train Loss @ epoch {epoch+1:04d} iter {iter_count+1:08d} = {loss.item():.6f}")

                iter_count += 1

        results['best_val_loss'].append(best_val_loss)
        pickle.dump(results, open(os.path.join(self.save_dir, 'train_stats_phase1.p'), 'wb'))
        logger.info(f"Phase1 Best Val Loss = {best_val_loss:.6f}")

        return best_val_loss

    # ==================== Phase 2 (Session III) ==================== #

    def train_phase2(self):
        """Freeze InnerNet, reinitialize and retrain outer network.
        Skipped for baseline models.
        """
        if not self.has_inner_net:
            print("Baseline model — skipping phase2.")
            self._mark_stage('PHASE2_DONE')
            return

        self.config.seed = self.seed

        if self.is_rnn:
            best_val_loss = self._train_phase2_rnn()
        else:
            best_val_loss = self._train_phase2_classification()

        self._mark_stage('PHASE2_DONE')
        return best_val_loss

    def _train_phase2_classification(self):
        """Phase 2 for classification: freeze InnerNet, retrain outer."""
        train_loader, val_loader = self._get_classification_loaders()

        model = eval(self.model_conf.name)(self.config)
        # Load InnerNet from phase1
        load_model(model.inner_net,
                   self.save_dir + self.train_conf.best_model)

        # Freeze InnerNet
        self._freeze_inner_net(model)

        model = model.to(self.device)
        if self.use_gpu and len(self.gpus) > 1:
            model = nn.DataParallel(model, device_ids=self.gpus)

        optimizer = self._make_optimizer(model)
        early_stop = EarlyStopper([0.0], win_size=self.train_conf.get('early_stop_window', 20), is_decrease=False)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.train_conf.lr_decay_steps,
            gamma=self.train_conf.lr_decay)

        iter_count = 0
        results = defaultdict(list)
        best_val_loss = float('inf')
        best_val_acc = 0.0

        phase2_max_epoch = self.train_phase2_conf.get('max_epoch', self.train_conf.max_epoch)
        for epoch in range(phase2_max_epoch):
            # Validation
            if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
                model.eval()
                val_loss_sum = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs = imgs.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        out, loss, _ = model(imgs, labels)
                        val_loss_sum += loss.item()
                        _, pred = torch.max(out, 1)
                        total += labels.size(0)
                        correct += (pred == labels).sum().item()

                avg_val_loss = val_loss_sum / len(val_loader)
                acc = correct / total
                results['val_loss'].append(avg_val_loss)
                results['val_acc'].append(acc)

                logger.info(f"Epoch {epoch+1} | Val Loss = {avg_val_loss:.6f} | Val Acc = {acc:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = acc
                    model_to_save = model.module if hasattr(model, 'module') else model
                    snapshot(model_to_save, optimizer, self.config,
                             epoch + 1, tag='best_phase2')

                if early_stop.tick([avg_val_loss]):
                    logger.info("Early stopping triggered.")
                    break

            # Training
            model.train()
            lr_scheduler.step()

            for imgs, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                _, loss, _ = model(imgs, labels)
                loss.backward()
                optimizer.step()

                results['train_loss'].append(loss.item())
                results['train_step'].append(iter_count)

                if (iter_count + 1) % self.train_conf.display_iter == 0:
                    logger.info(f"Train Loss @ epoch {epoch+1:04d} iter {iter_count+1:08d} = {loss.item():.6f}")

                iter_count += 1

        results['best_val_loss'].append(best_val_loss)
        results['best_val_acc'].append(best_val_acc)
        pickle.dump(results, open(os.path.join(self.save_dir, 'train_stats_phase2.p'), 'wb'))
        logger.info(f"Phase2 Best Val Loss = {best_val_loss:.6f} | Best Val Acc = {best_val_acc:.4f}")

        return best_val_loss

    def _train_phase2_rnn(self):
        """Phase 2 for RNN: freeze InnerNet, retrain outer."""
        train_data = self._batchify(self.corpus.train, self.train_conf.batch_size)
        val_data = self._batchify(self.corpus.valid, self.train_conf.batch_size)

        model = eval(self.model_conf.name)(self.config, self.ntokens)
        load_model(model.inner_net,
                   self.save_dir + self.train_conf.best_model)

        self._freeze_inner_net(model)

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        optimizer = self._make_optimizer(model)
        early_stop = EarlyStopper([0.0], win_size=self.train_conf.get('early_stop_window', 20), is_decrease=False)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.train_conf.lr_decay_steps,
            gamma=self.train_conf.lr_decay)

        iter_count = 0
        results = defaultdict(list)
        best_val_loss = np.inf

        phase2_max_epoch = self.train_phase2_conf.get('max_epoch', self.train_conf.max_epoch)
        for epoch in range(phase2_max_epoch):
            # Validation
            if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
                model.eval()
                val_loss = []
                hidden = None
                for i in range(0, val_data.size(0) - 1, self.train_conf.bptt):
                    data, targets = self._get_batch(val_data, i)
                    if self.use_gpu:
                        data, targets = data.cuda(), targets.cuda()
                    with torch.no_grad():
                        _, hidden, loss, _ = model(data, targets, mask=None, hx=hidden)
                    hidden = self._repackage_hidden(hidden)
                    val_loss.append(loss.item())

                avg_val_loss = np.mean(val_loss)
                results['val_loss'].append(avg_val_loss)
                logger.info(f"Epoch {epoch+1} | Val Loss = {avg_val_loss:.6f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model_to_save = model.module if hasattr(model, 'module') else model
                    snapshot(model_to_save, optimizer, self.config,
                             epoch + 1, tag='best_phase2')

                if early_stop.tick([avg_val_loss]):
                    logger.info("Early stopping triggered.")
                    break

            # Training
            model.train()
            lr_scheduler.step()
            hidden = None
            mask = [None] * len(self.model_conf.out_hidden_dim)
            for i in range(len(self.model_conf.out_hidden_dim)):
                mask[i] = (torch.bernoulli(
                    torch.Tensor(self.model_conf.out_hidden_dim[i]).fill_(
                        1 - self.model_conf.dropout))
                    / (1 - self.model_conf.dropout))

            for batch, i in enumerate(range(0, train_data.size(0) - 1, self.train_conf.bptt)):
                optimizer.zero_grad(set_to_none=True)
                data, targets = self._get_batch(train_data, i)
                if self.use_gpu:
                    data, targets = data.cuda(), targets.cuda()
                _, hidden, loss, _ = model(data, targets, mask=mask, hx=hidden)
                hidden = self._repackage_hidden(hidden)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.train_conf.clip)
                optimizer.step()

                results['train_loss'].append(loss.item())
                results['train_step'].append(iter_count)

                if (iter_count + 1) % self.train_conf.display_iter == 0:
                    logger.info(f"Train Loss @ epoch {epoch+1:04d} iter {iter_count+1:08d} = {loss.item():.6f}")

                iter_count += 1

        results['best_val_loss'].append(best_val_loss)
        pickle.dump(results, open(os.path.join(self.save_dir, 'train_stats_phase2.p'), 'wb'))
        logger.info(f"Phase2 Best Val Loss = {best_val_loss:.6f}")

        return best_val_loss

    # ==================== Test ==================== #

    def test(self):
        """Evaluate best model on test set."""
        if self.is_rnn:
            result = self._test_rnn()
        else:
            result = self._test_classification()

        self._mark_stage('TEST_DONE')
        self._mark_stage('COMPLETED')
        return result

    def _test_classification(self):
        """Test for classification tasks."""
        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            test_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                           train=False, transform=transform, download=True)
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
            test_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                             train=False, transform=transform, download=True)
        else:
            raise ValueError("Non-supported dataset!")

        test_loader = DataLoader(test_dataset, batch_size=self.test_conf.batch_size, shuffle=False)

        # Create and load model
        model = eval(self.model_conf.name)(self.config)
        if self.has_inner_net:
            test_model_path = self.save_dir + self.test_conf.test_model
        else:
            # Baseline: use phase1 model
            test_model_path = self.save_dir + '/model_snapshot_best_phase1.pth'
        load_model(model, test_model_path)

        model = model.to(self.device)
        if self.use_gpu and len(self.gpus) > 1:
            model = nn.DataParallel(model, device_ids=self.gpus)

        model.eval()
        correct = 0
        total = 0
        in2cells = []
        cnt = 0

        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.no_grad():
                out, _, in2cells_batch = model(imgs, labels, collect=self.has_inner_net)

                if self.has_inner_net:
                    for i, layer_data in enumerate(in2cells_batch):
                        if cnt == 0:
                            in2cells.append(layer_data)
                        elif cnt < 100:
                            in2cells[i] = np.concatenate((in2cells[i], layer_data), 0)
                cnt += 1

                _, pred = torch.max(out, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        test_accuracy = correct / total
        logger.info(f"Test Accuracy = {test_accuracy:.4f}")

        # Save results
        test_results = {'test_accuracy': test_accuracy}
        pickle.dump(test_results, open(os.path.join(self.save_dir, 'test_results.p'), 'wb'))
        if in2cells:
            pickle.dump(in2cells, open(os.path.join(self.save_dir, 'in2cells.p'), 'wb'))

        return test_accuracy

    def _test_rnn(self):
        """Test for RNN language modeling."""
        test_data = self._batchify(self.corpus.test, self.test_conf.batch_size)

        model = eval(self.model_conf.name)(self.config, self.ntokens)
        if self.has_inner_net:
            test_model_path = self.save_dir + self.test_conf.test_model
        else:
            test_model_path = self.save_dir + '/model_snapshot_best_phase1.pth'
        load_model(model, test_model_path)

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        model.eval()
        test_loss = []
        in2cells = []
        cnt = 0
        hidden = None

        for i in range(0, test_data.size(0) - 1, self.train_conf.bptt):
            data, targets = self._get_batch(test_data, i)
            if self.use_gpu:
                data, targets = data.cuda(), targets.cuda()
            with torch.no_grad():
                _, hidden, loss, in2cells_batch = model(data, targets, mask=None, hx=hidden)

                if self.has_inner_net:
                    for j, layer_data in enumerate(in2cells_batch):
                        if cnt == 0:
                            in2cells.append(layer_data)
                        else:
                            in2cells[j] = np.concatenate((in2cells[j], layer_data), 0)
                cnt += 1

                hidden = self._repackage_hidden(hidden)
                test_loss.append(loss.item())

        avg_test_loss = np.mean(test_loss)
        logger.info(f"Test Loss = {avg_test_loss:.6f}")

        test_results = {'test_loss': avg_test_loss}
        pickle.dump(test_results, open(os.path.join(self.save_dir, 'test_results.p'), 'wb'))
        if in2cells:
            pickle.dump(in2cells, open(os.path.join(self.save_dir, 'in2cells.p'), 'wb'))

        return avg_test_loss

    # ==================== Helper Methods ==================== #

    def _get_classification_loaders(self):
        """Create train/val DataLoaders for classification tasks."""
        loader_kwargs = {
            'batch_size': self.train_conf.batch_size,
            'num_workers': self.train_conf.get('num_workers', 0),
            'pin_memory': self.use_gpu and self.train_conf.get('num_workers', 0) > 0,
            'persistent_workers': self.train_conf.get('num_workers', 0) > 0
        }

        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            full_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                           train=True, transform=transform, download=True)
            train_dataset, val_dataset = random_split(full_dataset, [50000, 10000])
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
            full_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                             train=True, transform=transform, download=True)
            train_dataset, val_dataset = random_split(full_dataset, [40000, 10000])
        else:
            raise ValueError("Non-supported dataset!")

        train_loader = DataLoader(train_dataset, shuffle=self.train_conf.shuffle, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        return train_loader, val_loader

    def _make_optimizer(self, model):
        """Create optimizer from config."""
        params = [p for p in model.parameters() if p.requires_grad]
        if self.train_conf.optimizer == 'Adam':
            return optim.Adam(params, lr=self.train_conf.lr,
                               weight_decay=self.train_conf.wd)
        elif self.train_conf.optimizer == 'SGD':
            return optim.SGD(params, lr=self.train_conf.lr,
                              momentum=self.train_conf.momentum,
                              weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

    def _freeze_inner_net(self, model):
        """Freeze InnerNet parameters."""
        if hasattr(model, 'inner_net'):
            for param in model.inner_net.parameters():
                param.requires_grad = False

    def _mark_stage(self, stage_name):
        """Write a stage marker file for checkpoint resumption."""
        marker_path = os.path.join(self.save_dir, stage_name)
        with open(marker_path, 'w') as f:
            f.write('')

    def _batchify(self, data, batch_size):
        seq_len = data.size(0) // batch_size
        data = data.narrow(0, 0, seq_len * batch_size)
        data = data.view(batch_size, -1).t().contiguous()
        return data

    def _repackage_hidden(self, h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return [self._repackage_hidden(v) for v in h]

    def _get_batch(self, source, i):
        seq_len = min(self.train_conf.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target
