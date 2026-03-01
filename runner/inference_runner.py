from __future__ import (division, print_function)
import os
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from model import *
from dataset import *
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, load_model_v3, snapshot, load_model, EarlyStopper
from utils.corpus import Corpus

logger = get_logger('exp_logger')
EPS = float(np.finfo(np.float32).eps)
__all__ = ['XorNeuronRunner', 'XorNeuronLMRunner',
           'XorNeuronRunner_v2',
           'XorNeuronRunner_test']

class XorNeuronRunner_v2(object):

    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        self.num_cell_types = config.model.num_cell_types
        self.dataset_conf = config.dataset
        self.model_conf = config.model
        self.pretrain_conf = config.pretrain
        self.train_conf = config.train
        self.test_conf = config.test
        self.use_gpu = config.use_gpu
        self.gpus = config.gpus
        self.writer = SummaryWriter(config.save_dir)

    def pretrain(self, cell_type):
        self.config.seed = int(str(self.seed) + str(cell_type))
        train_dataset = eval(self.dataset_conf.loader_name)(self.config, split='train')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.pretrain_conf.batch_size,
            shuffle=self.pretrain_conf.shuffle,
            num_workers=self.pretrain_conf.num_workers,
            collate_fn=train_dataset.collate_fn,
            drop_last=False)

        # Train innernet
        model = InnerNet(self.config)
        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.pretrain_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.pretrain_conf.lr,
                momentum=self.pretrain_conf.momentum,
                weight_decay=self.pretrain_conf.wd)
        elif self.pretrain_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.pretrain_conf.lr,
                weight_decay=self.pretrain_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        # reset gradient
        optimizer.zero_grad()
        best_train_loss = np.inf
        for epoch in range(self.pretrain_conf.max_epoch):
            train_loss = []
            model.train()
            for xy, targets in train_loader:
                optimizer.zero_grad()

                if self.use_gpu:
                    xy, targets = xy.cuda(), targets.cuda()

                _, loss = model(xy, targets)
                loss.backward()
                optimizer.step()

                # train_loss += [float(loss.data.cpu().numpy())]
                train_loss.append(loss.item())

            # display loss
            train_loss = np.stack(train_loss).mean()

            # save best model
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                print("Pretrain Loss @ epoch {:04d} = {}".format(epoch + 1, np.mean(best_train_loss)))
                snapshot(
                    model.module.inner_net if self.use_gpu else model.inner_net,
                    optimizer,
                    self.config,
                    epoch + 1,
                    tag='best_pretrained' + str(cell_type))

        return 1

    def train_phase1_v2(self):
        # 0. 设置种子
        self.config.seed = self.seed
        
        # ========================= 1. 数据集准备 (优化版) ========================= #
        # 定义通用加载参数，开启 pin_memory 和多线程
        loader_kwargs = {
            'batch_size': self.train_conf.batch_size,
            'num_workers': 4,  # 🚀 开启4个子进程加载数据
            'pin_memory': True if self.use_gpu else False,  # 🚀 锁页内存，加速 CPU->GPU 传输
            'persistent_workers': True if self.train_conf.batch_size > 0 else False # 避免每个epoch重新创建线程
        }

        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            full_dataset = datasets.MNIST(root=self.dataset_conf.data_path, train=True, transform=transform, download=True)
            train_dataset, val_dataset = random_split(full_dataset, [50000, 10000])
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
            full_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path, train=True, transform=transform, download=True)
            train_dataset, val_dataset = random_split(full_dataset, [40000, 10000])
        else:
            raise ValueError("Non-supported dataset!")

        train_loader = DataLoader(train_dataset, shuffle=self.train_conf.shuffle, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

        # ========================= 2. 模型与优化器初始化 ========================= #
        model = eval(self.model_conf.name)(self.config)
        print(self.model_conf.name.lower())
        # 🚀 修正加载逻辑：检测是否为 V3 架构
        if 'v3' in self.model_conf.name.lower():
            # 使用特殊的拼接加载函数
            load_model_v3(model, self.pretrain_conf.best_model, self.config.save_dir)
        else:
            # 原有的 V1/V2 循环加载逻辑
            for i in range(self.num_cell_types):
                load_model(model.inner_net[i], self.config.save_dir + self.pretrain_conf.best_model[i])

        # 设备处理
        device = torch.device('cuda' if self.use_gpu else 'cpu')
        model = model.to(device)
        
        if self.use_gpu and len(self.gpus) > 1:
            model = nn.DataParallel(model, device_ids=self.gpus)

        # 优化器
        params = [p for p in model.parameters() if p.requires_grad]
        opt_conf = self.train_conf
        if opt_conf.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=opt_conf.lr, momentum=opt_conf.momentum, weight_decay=opt_conf.wd)
        elif opt_conf.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=opt_conf.lr, weight_decay=opt_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt_conf.lr_decay_steps, gamma=opt_conf.lr_decay)

        # ========================= 3. 极速训练循环 ========================= #
        iter_count = 0
        results = defaultdict(list)
        best_val_loss = float('inf')
        
        # 预先定义好常量，减少循环内属性查找
        display_iter = opt_conf.display_iter
        snapshot_epoch = opt_conf.snapshot_epoch
        valid_epoch = opt_conf.valid_epoch

        for epoch in range(opt_conf.max_epoch):
            # --------------------- Validation (优化版) --------------------- #
            if (epoch + 1) % valid_epoch == 0 or epoch == 0:
                model.eval()
                val_loss_sum = 0.0
                correct = 0
                total = 0
                
                # 🚀 不计算梯度，不仅省显存，还快
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        if self.use_gpu:
                            imgs = imgs.to(device, non_blocking=True)
                            labels = labels.to(device, non_blocking=True)

                        out, loss, _ = model(imgs, labels)
                        
                        # 🚀 .item() 直接获取数值，不产生 Tensor 开销
                        val_loss_sum += loss.item()
                        
                        _, pred = torch.max(out, 1) # out.data -> out
                        total += labels.size(0)
                        correct += (pred == labels).sum().item()

                # 计算平均值
                avg_val_loss = val_loss_sum / len(val_loader)
                acc = correct / total

                # 记录结果
                results['val_loss'].append(avg_val_loss)
                results['val_acc'].append(acc)
                
                logger.info(f"Avg. Validation Loss = {avg_val_loss:.6f} +- 0")
                self.writer.add_scalar('val_loss', avg_val_loss, iter_count)

                # Save Best
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = acc
                    # 获取原始模型 (处理 DataParallel)
                    model_to_save = model.module.inner_net if hasattr(model, 'module') else model.inner_net
                    snapshot(model_to_save, optimizer, self.config, epoch + 1, tag='best_phase1')
                
                logger.info(f"Current Best Validation Loss = {best_val_loss:.6f}")

                # Early Stop Check
                if early_stop.tick([avg_val_loss]):
                    model_to_save = model.module if hasattr(model, 'module') else model
                    snapshot(model_to_save, optimizer, self.config, epoch + 1, tag='last')
                    self.writer.close()
                    break

            # --------------------- Training (优化版) --------------------- #
            model.train()
            lr_scheduler.step()
            
            for imgs, labels in train_loader:
                # 🚀 set_to_none=True 比 =0 更快
                optimizer.zero_grad(set_to_none=True)
                
                if self.use_gpu:
                    # 🚀 non_blocking=True 实现数据传输与计算流水线并行
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                _, loss, _ = model(imgs, labels)
                loss.backward()
                optimizer.step()

                # 🚀 .item() 提速
                loss_val = loss.item()
                
                results['train_loss'].append(loss_val)
                results['train_step'].append(iter_count)
                self.writer.add_scalar('train_loss', loss_val, iter_count)

                if (iter_count + 1) % display_iter == 0:
                    logger.info(f"Train Loss @ epoch {epoch + 1:04d} iteration {iter_count + 1:08d} = {loss_val:.6f}")

                iter_count += 1

            # Snapshot (Epoch End)
            if (epoch + 1) % snapshot_epoch == 0:
                logger.info(f"Saving Snapshot @ epoch {epoch + 1:04d}")
                model_to_save = model.module if hasattr(model, 'module') else model
                snapshot(model_to_save, optimizer, self.config, epoch + 1)

        # --------------------- Final Wrap up --------------------- #
        results['best_val_loss'].append(best_val_loss)
        results['best_val_acc'].append(best_val_acc) # type: ignore
        pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats_phase1.p'), 'wb'))
        self.writer.close()
        logger.info(f"Best Validation Loss = {best_val_loss:.6f}")

        return best_val_loss
    
    def train_phase2(self):
        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            train_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                           train=True,
                                           transform=transform,
                                           download=True)
            train_dataset, val_dataset = random_split(train_dataset, [50000, 10000])
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            train_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                             train=True,
                                             transform=transform,
                                             download=True)
            train_dataset, val_dataset = random_split(train_dataset, [40000, 10000])
        else:
            raise ValueError("Non-supported dataset!")

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.train_conf.batch_size,
                                  shuffle=self.train_conf.shuffle)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.train_conf.batch_size,
                                shuffle=False)

        # create models
        model = eval(self.model_conf.name)(self.config)
        # load inner-net trained on phase 1
        load_model(model.inner_net, self.config.save_dir + self.train_conf.best_model)

        # ====== Freeze InnerNet  ====== #
        for child in model.children():
            if isinstance(child, nn.ModuleList):
                for ch in child.children():
                    if isinstance(ch, nn.Sequential):  # InnerNet must be the only child built on nn.Sequential()
                        for param in ch.parameters():
                            param.requires_grad = False

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum,
                weight_decay=self.train_conf.wd)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_conf.lr_decay_steps,
            gamma=self.train_conf.lr_decay)

        # reset gradient
        optimizer.zero_grad()

        # resume training
        if self.train_conf.is_resume:
            load_model(model, self.train_conf.resume_model, optimizer=optimizer)

        # ========================= Training Loop ============================= #
        iter_count = 0
        results = defaultdict(list)
        best_val_loss = np.inf
        for epoch in range(self.train_conf.max_epoch):
            # ===================== validation ============================ #
            if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
                model.eval()
                val_loss = []
                correct = 0
                total = 0
                for imgs, labels in tqdm(val_loader):
                    if self.use_gpu:
                        imgs, labels = imgs.cuda(), labels.cuda()

                    with torch.no_grad():
                        out, loss, _ = model(imgs, labels)
                    # val_loss += [float(loss.data.cpu().numpy())]
                    val_loss.append(loss.item())

                    _, pred = torch.max(out.data, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

                val_loss = np.stack(val_loss).mean()
                results['val_loss'] += [val_loss]
                results['val_acc'] += [correct / total]
                logger.info("Avg. Validation Loss = {} +- {}".format(val_loss, 0))

                self.writer.add_scalar('val_loss', val_loss, iter_count)

                # save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = correct / total
                    snapshot(
                        model.module if self.use_gpu else model,
                        optimizer,
                        self.config,
                        epoch + 1,
                        tag='best_phase2')

                logger.info("Current Best Validation Loss = {}".format(best_val_loss))

                # check early stop
                if early_stop.tick([val_loss]):
                    snapshot(
                        model.module if self.use_gpu else model,
                        optimizer,
                        self.config,
                        epoch + 1,
                        tag='last')
                    self.writer.close()
                    break
            # ====================== training ============================= #
            model.train()
            lr_scheduler.step()
            for imgs, labels in train_loader:
                # 0. clears all gradients.
                optimizer.zero_grad()
                if self.use_gpu:
                    imgs, labels = imgs.cuda(), labels.cuda()

                # 1. forward pass
                # 2. compute loss
                _, loss, _ = model(imgs, labels)
                # 3. backward pass (accumulates gradients).
                loss.backward()
                # 4. performs a single update step.
                optimizer.step()

                # train_loss = float(loss.data.cpu().numpy())
                train_loss = loss.item()
                results['train_loss'] += [train_loss]
                results['train_step'] += [iter_count]
                self.writer.add_scalar('train_loss', train_loss, iter_count)

                # display loss
                if (iter_count + 1) % self.train_conf.display_iter == 0:
                    logger.info(
                        "Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

                iter_count += 1

            # snapshot model
            if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
                logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
                snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1)

        results['best_val_loss'] += [best_val_loss]
        results['best_val_acc'] += [best_val_acc]
        pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats_phase2.p'), 'wb'))
        self.writer.close()
        logger.info("Best Validation Loss = {}".format(best_val_loss))

        return best_val_loss

    def test(self):
        print(self.dataset_conf.loader_name)
        print(self.dataset_conf.split)
        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            test_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                          train=False,
                                          transform=transform,
                                          download=True)
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            test_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                            train=False,
                                            transform=transform,
                                            download=True)
        else:
            raise ValueError("Non-supported dataset!")

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.test_conf.batch_size,
                                 shuffle=False)

        # # create models
        # model = eval(self.model_conf.name)(self.config)
        # if 'xor_neuron' in self.model_conf.name:
        #   load_model(model, self.test_conf.test_model)

        # create models
        model = eval(self.model_conf.name)(self.config)
        # load test model
        print(self.config.save_dir)
        # load_model(model, self.test_conf.test_model)
        load_model(model, self.config.save_dir + self.test_conf.test_model)

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        model.eval()
        correct = 0
        total = 0
        in2cells = []
        cnt = 0
        for imgs, labels in tqdm(test_loader):
            if self.use_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()

            with torch.no_grad():
                out, _, in2cells_per_layer_per_batch = model(imgs, labels, collect=True)

                # in2cells : num_layers x [data_size x num_cell_types x ... x arity]
                for i, in2cells_per_layer in enumerate(in2cells_per_layer_per_batch):
                    if cnt == 0:
                        in2cells.append(in2cells_per_layer)
                    elif cnt < 100 and self.dataset_conf.name == 'mnist':
                        in2cells[i] = np.concatenate((in2cells[i], in2cells_per_layer), 0)
                cnt += 1

                _, pred = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        test_accuracy = correct / total
        logger.info("Test Accuracy = {} +- {}".format(test_accuracy, 0))
        pickle.dump(in2cells, open(os.path.join(self.config.save_dir, 'in2cells.p'), 'wb'))

        return test_accuracy

    def test_local(self):
        print(self.dataset_conf.loader_name)
        print(self.dataset_conf.split)
        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            test_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                          train=False,
                                          transform=transform,
                                          download=True)
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            test_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                            train=False,
                                            transform=transform,
                                            download=True)
        else:
            raise ValueError("Non-supported dataset!")

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.test_conf.batch_size,
                                 shuffle=False)

        # # create models
        # model = eval(self.model_conf.name)(self.config)
        # if 'xor_neuron' in self.model_conf.name:
        #   load_model(model, self.test_conf.test_model)

        # create models
        model = eval(self.model_conf.name + '_test')(self.config)
        # load test model
        print(self.config.save_dir)
        load_model(model, self.test_conf.test_model)

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        model.eval()
        correct = 0
        total = 0
        input2innerAll = {}
        for imgs, labels in tqdm(test_loader):
            if self.use_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()

            with torch.no_grad():
                out, _, input2inner = model(imgs, labels)

                for i in range(len(input2inner)):
                    if len(input2innerAll.keys()) < len(input2inner):
                        input2innerAll[i] = input2inner[i]
                    else:
                        input2innerAll[i] = np.vstack((input2innerAll[i], input2inner[i]))

                _, pred = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        test_accuracy = correct / total
        logger.info("Test Accuracy = {} +- {}".format(test_accuracy, 0))
        pickle.dump(input2innerAll, open(os.path.join(self.config.save_dir, 'input2innerAll.p'), 'wb'))
        print("Saved input2innerAll.p to {}".format(self.config.save_dir))

        return test_accuracy




class XorNeuronRunner_test(object):

    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        self.num_cell_types = config.model.num_cell_types
        self.dataset_conf = config.dataset
        self.model_conf = config.model
        self.pretrain_conf = config.pretrain
        self.train_conf = config.train
        self.test_conf = config.test
        self.use_gpu = config.use_gpu
        self.gpus = config.gpus
        self.writer = SummaryWriter(config.save_dir)

    def pretrain(self, cell_type):
        self.config.seed = int(str(self.seed) + str(cell_type))
        train_dataset = eval(self.dataset_conf.loader_name)(self.config, split='train')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.pretrain_conf.batch_size,
            shuffle=self.pretrain_conf.shuffle,
            num_workers=self.pretrain_conf.num_workers,
            collate_fn=train_dataset.collate_fn,
            drop_last=False)

        # Train innernet
        model = InnerNet(self.config)
        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.pretrain_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.pretrain_conf.lr,
                momentum=self.pretrain_conf.momentum,
                weight_decay=self.pretrain_conf.wd)
        elif self.pretrain_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.pretrain_conf.lr,
                weight_decay=self.pretrain_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        # reset gradient
        optimizer.zero_grad()
        best_train_loss = np.inf
        for epoch in range(self.pretrain_conf.max_epoch):
            train_loss = []
            model.train()
            for xy, targets in train_loader:
                optimizer.zero_grad()

                if self.use_gpu:
                    xy, targets = xy.cuda(), targets.cuda()

                _, loss = model(xy, targets)
                loss.backward()
                optimizer.step()

                # train_loss += [float(loss.data.cpu().numpy())]
                train_loss.append(loss.item())

            # display loss
            train_loss = np.stack(train_loss).mean()

            # save best model
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                print("Pretrain Loss @ epoch {:04d} = {}".format(epoch + 1, np.mean(best_train_loss)))
                snapshot(
                    model.module.inner_net if self.use_gpu else model.inner_net,
                    optimizer,
                    self.config,
                    epoch + 1,
                    tag='best_pretrained' + str(cell_type))

        return 1

    def train_phase1(self):
        # prepare data
        self.config.seed = self.seed
        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            train_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                           train=True,
                                           transform=transform,
                                           download=True)
            train_dataset, val_dataset = random_split(train_dataset, [50000, 10000])
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            train_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                             train=True,
                                             transform=transform,
                                             download=True)
            train_dataset, val_dataset = random_split(train_dataset, [40000, 10000])
        else:
            raise ValueError("Non-supported dataset!")

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.train_conf.batch_size,
                                  shuffle=self.train_conf.shuffle)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.train_conf.batch_size,
                                shuffle=False)

        # create models
        model = eval(self.model_conf.name)(self.config)
        if isinstance(model, nn.DataParallel):
            target_inner_net = model.module.inner_net
        else:
            target_inner_net = model.inner_net

        # load pretrained inner-net
        load_model(target_inner_net, self.config.save_dir + self.pretrain_conf.best_model[0])
        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum,
                weight_decay=self.train_conf.wd)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_conf.lr_decay_steps,
            gamma=self.train_conf.lr_decay)

        # reset gradient
        optimizer.zero_grad()

        # resume training
        load_model(model.inner_net, self.config.save_dir + self.pretrain_conf.best_model[0])

        # ========================= Training Loop ============================= #
        iter_count = 0
        results = defaultdict(list)
        best_val_loss = np.inf

        # main loop
        for epoch in range(self.train_conf.max_epoch):
            # ===================== validation ============================ #
            if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
                model.eval()
                val_loss = []
                correct = 0
                total = 0
                for imgs, labels in tqdm(val_loader):
                    if self.use_gpu:
                        imgs, labels = imgs.cuda(), labels.cuda()

                    with torch.no_grad():
                        out, loss, _ = model(imgs, labels)
                    # val_loss += [float(loss.data.cpu().numpy())]
                    val_loss.append(loss.item())

                    _, pred = torch.max(out.data, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

                val_loss = np.stack(val_loss).mean()
                results['val_loss'] += [val_loss]
                results['val_acc'] += [correct / total]
                logger.info("Avg. Validation Loss = {} +- {}".format(val_loss, 0))

                self.writer.add_scalar('val_loss', val_loss, iter_count)

                # save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = correct / total
                    snapshot(
                        model.module.inner_net if self.use_gpu else model.inner_net,
                        optimizer,
                        self.config,
                        epoch + 1,
                        tag='best_phase1')

                logger.info("Current Best Validation Loss = {}".format(best_val_loss))

                from scipy.signal import convolve2d
                from scipy.stats import multivariate_normal
                import matplotlib.pyplot as plt
                nb = 101
                x = np.linspace(-5, 5, nb)
                y = np.linspace(-5, 5, nb)
                xv, yv = np.meshgrid(x, y)
                xy = np.vstack([xv.reshape(-1), yv.reshape(-1)]).T
                mvn = multivariate_normal(mean=[0, 0], cov=[[1/9, 0], [0, 1/9]])
                gaussian_kernel = mvn.pdf(xy).reshape(nb, nb)
                gaussian_kernel /= gaussian_kernel.sum()
                npr = np.random.RandomState(seed=self.config.seed)
                init_unif = npr.uniform(-1, 1, size=(nb, nb))
                targets = convolve2d(init_unif, gaussian_kernel, mode='same').reshape(-1, 1)
                out_phase1 = model.inner_net[0](torch.Tensor(xy))
                plt.imshow(targets.reshape(101, 101), cmap='bwr')
                plt.figure()
                plt.imshow(out_phase1.data.cpu().numpy().reshape(101, 101), cmap='bwr')
                plt.title("{}".format(epoch))
                plt.show()
                import pdb
                pdb.set_trace()

                # check early stop
                if early_stop.tick([val_loss]):
                    snapshot(
                        model.module if self.use_gpu else model,
                        optimizer,
                        self.config,
                        epoch + 1,
                        tag='last')
                    self.writer.close()
                    break
            # ====================== training ============================= #
            model.train()
            lr_scheduler.step()
            for imgs, labels in train_loader:
                # 0. clears all gradients.
                optimizer.zero_grad()
                if self.use_gpu:
                    imgs, labels = imgs.cuda(), labels.cuda()

                # 1. forward pass
                # 2. compute loss
                _, loss, _ = model(imgs, labels)
                # 3. backward pass (accumulates gradients).
                loss.backward()
                # 4. performs a single update step.
                optimizer.step()

                # train_loss = float(loss.data.cpu().numpy())
                train_loss=loss.item()
                results['train_loss'] += [train_loss]
                results['train_step'] += [iter_count]
                self.writer.add_scalar('train_loss', train_loss, iter_count)

                # display loss
                if (iter_count + 1) % self.train_conf.display_iter == 0:
                    logger.info(
                        "Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

                iter_count += 1

            # snapshot model
            if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
                logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
                snapshot(model.module.inner_net if self.use_gpu else model.inner_net, optimizer, self.config, epoch + 1)

        results['best_val_loss'] += [best_val_loss]
        results['best_val_acc'] += [best_val_acc]
        pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats_phase1.p'), 'wb'))
        self.writer.close()
        logger.info("Best Validation Loss = {}".format(best_val_loss))

        return best_val_loss

    def train_phase1_v2(self):
        # 0. 设置种子
        self.config.seed = self.seed
        
        # ========================= 1. 数据集准备 (优化版) ========================= #
        # 定义通用加载参数，开启 pin_memory 和多线程
        loader_kwargs = {
            'batch_size': self.train_conf.batch_size,
            'num_workers': 4,  # 🚀 开启4个子进程加载数据
            'pin_memory': True if self.use_gpu else False,  # 🚀 锁页内存，加速 CPU->GPU 传输
            'persistent_workers': True if self.train_conf.batch_size > 0 else False # 避免每个epoch重新创建线程
        }

        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            full_dataset = datasets.MNIST(root=self.dataset_conf.data_path, train=True, transform=transform, download=True)
            train_dataset, val_dataset = random_split(full_dataset, [50000, 10000])
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
            full_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path, train=True, transform=transform, download=True)
            train_dataset, val_dataset = random_split(full_dataset, [40000, 10000])
        else:
            raise ValueError("Non-supported dataset!")

        train_loader = DataLoader(train_dataset, shuffle=self.train_conf.shuffle, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

        # ========================= 2. 模型与优化器初始化 ========================= #
        model = eval(self.model_conf.name)(self.config)
        
        # 加载预训练 InnerNet
        load_model(model.inner_net, self.config.save_dir + self.pretrain_conf.best_model[0])
        # 设备处理
        device = torch.device('cuda' if self.use_gpu else 'cpu')
        model = model.to(device)
        
        if self.use_gpu and len(self.gpus) > 1:
            model = nn.DataParallel(model, device_ids=self.gpus)

        # 优化器
        params = [p for p in model.parameters() if p.requires_grad]
        opt_conf = self.train_conf
        if opt_conf.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=opt_conf.lr, momentum=opt_conf.momentum, weight_decay=opt_conf.wd)
        elif opt_conf.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=opt_conf.lr, weight_decay=opt_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt_conf.lr_decay_steps, gamma=opt_conf.lr_decay)

        # ========================= 3. 极速训练循环 ========================= #
        iter_count = 0
        results = defaultdict(list)
        best_val_loss = float('inf')
        
        # 预先定义好常量，减少循环内属性查找
        display_iter = opt_conf.display_iter
        snapshot_epoch = opt_conf.snapshot_epoch
        valid_epoch = opt_conf.valid_epoch

        for epoch in range(opt_conf.max_epoch):
            # --------------------- Validation (优化版) --------------------- #
            if (epoch + 1) % valid_epoch == 0 or epoch == 0:
                model.eval()
                val_loss_sum = 0.0
                correct = 0
                total = 0
                
                # 🚀 不计算梯度，不仅省显存，还快
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        if self.use_gpu:
                            imgs = imgs.to(device, non_blocking=True)
                            labels = labels.to(device, non_blocking=True)

                        out, loss, _ = model(imgs, labels)
                        
                        # 🚀 .item() 直接获取数值，不产生 Tensor 开销
                        val_loss_sum += loss.item()
                        
                        _, pred = torch.max(out, 1) # out.data -> out
                        total += labels.size(0)
                        correct += (pred == labels).sum().item()

                # 计算平均值
                avg_val_loss = val_loss_sum / len(val_loader)
                acc = correct / total

                # 记录结果
                results['val_loss'].append(avg_val_loss)
                results['val_acc'].append(acc)
                
                logger.info(f"Avg. Validation Loss = {avg_val_loss:.6f} +- 0")
                self.writer.add_scalar('val_loss', avg_val_loss, iter_count)

                # Save Best
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = acc
                    # 获取原始模型 (处理 DataParallel)
                    model_to_save = model.module.inner_net if hasattr(model, 'module') else model.inner_net
                    snapshot(model_to_save, optimizer, self.config, epoch + 1, tag='best_phase1')
                
                logger.info(f"Current Best Validation Loss = {best_val_loss:.6f}")

                # Early Stop Check
                if early_stop.tick([avg_val_loss]):
                    model_to_save = model.module if hasattr(model, 'module') else model
                    snapshot(model_to_save, optimizer, self.config, epoch + 1, tag='last')
                    self.writer.close()
                    break

            # --------------------- Training (优化版) --------------------- #
            model.train()
            lr_scheduler.step()
            
            for imgs, labels in train_loader:
                # 🚀 set_to_none=True 比 =0 更快
                optimizer.zero_grad(set_to_none=True)
                
                if self.use_gpu:
                    # 🚀 non_blocking=True 实现数据传输与计算流水线并行
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                _, loss, _ = model(imgs, labels)
                loss.backward()
                optimizer.step()

                # 🚀 .item() 提速
                loss_val = loss.item()
                
                results['train_loss'].append(loss_val)
                results['train_step'].append(iter_count)
                self.writer.add_scalar('train_loss', loss_val, iter_count)

                if (iter_count + 1) % display_iter == 0:
                    logger.info(f"Train Loss @ epoch {epoch + 1:04d} iteration {iter_count + 1:08d} = {loss_val:.6f}")

                iter_count += 1

            # Snapshot (Epoch End)
            if (epoch + 1) % snapshot_epoch == 0:
                logger.info(f"Saving Snapshot @ epoch {epoch + 1:04d}")
                model_to_save = model.module if hasattr(model, 'module') else model
                snapshot(model_to_save, optimizer, self.config, epoch + 1)

            # --------------------- Final Wrap up --------------------- #
            results['best_val_loss'].append(best_val_loss)
            results['best_val_acc'].append(best_val_acc) # type: ignore
            pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats_phase1.p'), 'wb'))
            self.writer.close()
            logger.info(f"Best Validation Loss = {best_val_loss:.6f}")

        return best_val_loss
    
    def train_phase2(self):
        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            train_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                           train=True,
                                           transform=transform,
                                           download=True)
            train_dataset, val_dataset = random_split(train_dataset, [50000, 10000])
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            train_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                             train=True,
                                             transform=transform,
                                             download=True)
            train_dataset, val_dataset = random_split(train_dataset, [40000, 10000])
        else:
            raise ValueError("Non-supported dataset!")

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.train_conf.batch_size,
                                  shuffle=self.train_conf.shuffle)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.train_conf.batch_size,
                                shuffle=False)

        # create models
        model = eval(self.model_conf.name)(self.config)
        # load inner-net trained on phase 1
        load_model(model.inner_net, self.config.save_dir + self.train_conf.best_model)

        # ====== Freeze InnerNet  ====== #
        for child in model.children():
            if isinstance(child, nn.ModuleList):
                for ch in child.children():
                    if isinstance(ch, nn.Sequential):  # InnerNet must be the only child built on nn.Sequential()
                        for param in ch.parameters():
                            param.requires_grad = False

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum,
                weight_decay=self.train_conf.wd)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_conf.lr_decay_steps,
            gamma=self.train_conf.lr_decay)

        # reset gradient
        optimizer.zero_grad()

        # resume training
        if self.train_conf.is_resume:
            load_model(model, self.train_conf.resume_model, optimizer=optimizer)

        # ========================= Training Loop ============================= #
        iter_count = 0
        results = defaultdict(list)
        best_val_loss = np.inf
        for epoch in range(self.train_conf.max_epoch):
            # ===================== validation ============================ #
            if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
                model.eval()
                val_loss = []
                correct = 0
                total = 0
                for imgs, labels in tqdm(val_loader):
                    if self.use_gpu:
                        imgs, labels = imgs.cuda(), labels.cuda()

                    with torch.no_grad():
                        out, loss, _ = model(imgs, labels)
                    # val_loss += [float(loss.data.cpu().numpy())]
                    val_loss.append(loss.item())

                    _, pred = torch.max(out.data, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

                val_loss = np.stack(val_loss).mean()
                results['val_loss'] += [val_loss]
                results['val_acc'] += [correct / total]
                logger.info("Avg. Validation Loss = {} +- {}".format(val_loss, 0))

                self.writer.add_scalar('val_loss', val_loss, iter_count)

                # save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = correct / total
                    snapshot(
                        model.module if self.use_gpu else model,
                        optimizer,
                        self.config,
                        epoch + 1,
                        tag='best_phase2')

                logger.info("Current Best Validation Loss = {}".format(best_val_loss))

                # check early stop
                if early_stop.tick([val_loss]):
                    snapshot(
                        model.module if self.use_gpu else model,
                        optimizer,
                        self.config,
                        epoch + 1,
                        tag='last')
                    self.writer.close()
                    break
            # ====================== training ============================= #
            model.train()
            lr_scheduler.step()
            for imgs, labels in train_loader:
                # 0. clears all gradients.
                optimizer.zero_grad()
                if self.use_gpu:
                    imgs, labels = imgs.cuda(), labels.cuda()

                # 1. forward pass
                # 2. compute loss
                _, loss, _ = model(imgs, labels)
                # 3. backward pass (accumulates gradients).
                loss.backward()
                # 4. performs a single update step.
                optimizer.step()

                # train_loss = float(loss.data.cpu().numpy())
                train_loss = loss.item()
                results['train_loss'] += [train_loss]
                results['train_step'] += [iter_count]
                self.writer.add_scalar('train_loss', train_loss, iter_count)

                # display loss
                if (iter_count + 1) % self.train_conf.display_iter == 0:
                    logger.info(
                        "Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

                iter_count += 1

            # snapshot model
            if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
                logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
                snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1)

            results['best_val_loss'] += [best_val_loss]
            results['best_val_acc'] += [best_val_acc]
            pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats_phase2.p'), 'wb'))
            self.writer.close()
            logger.info("Best Validation Loss = {}".format(best_val_loss))

        return best_val_loss

    def test(self):
        print(self.dataset_conf.loader_name)
        print(self.dataset_conf.split)
        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            test_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                          train=False,
                                          transform=transform,
                                          download=True)
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            test_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                            train=False,
                                            transform=transform,
                                            download=True)
        else:
            raise ValueError("Non-supported dataset!")

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.test_conf.batch_size,
                                 shuffle=False)

        # # create models
        # model = eval(self.model_conf.name)(self.config)
        # if 'xor_neuron' in self.model_conf.name:
        #   load_model(model, self.test_conf.test_model)

        # create models
        model = eval(self.model_conf.name)(self.config)
        # load test model
        print(self.config.save_dir)
        # load_model(model, self.test_conf.test_model)
        load_model(model, self.config.save_dir + self.test_conf.test_model)

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        model.eval()
        correct = 0
        total = 0
        in2cells = []
        cnt = 0
        for imgs, labels in tqdm(test_loader):
            if self.use_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()

            with torch.no_grad():
                out, _, in2cells_per_layer_per_batch = model(imgs, labels, collect=True)

                # in2cells : num_layers x [data_size x num_cell_types x ... x arity]
                for i, in2cells_per_layer in enumerate(in2cells_per_layer_per_batch):
                    if cnt == 0:
                        in2cells.append(in2cells_per_layer)
                    elif cnt < 100 and self.dataset_conf.name == 'mnist':
                        in2cells[i] = np.concatenate((in2cells[i], in2cells_per_layer), 0)
                cnt += 1

                _, pred = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        test_accuracy = correct / total
        logger.info("Test Accuracy = {} +- {}".format(test_accuracy, 0))
        pickle.dump(in2cells, open(os.path.join(self.config.save_dir, 'in2cells.p'), 'wb'))

        return test_accuracy

    def test_local(self):
        print(self.dataset_conf.loader_name)
        print(self.dataset_conf.split)
        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            test_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                          train=False,
                                          transform=transform,
                                          download=True)
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            test_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                            train=False,
                                            transform=transform,
                                            download=True)
        else:
            raise ValueError("Non-supported dataset!")

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.test_conf.batch_size,
                                 shuffle=False)

        # # create models
        # model = eval(self.model_conf.name)(self.config)
        # if 'xor_neuron' in self.model_conf.name:
        #   load_model(model, self.test_conf.test_model)

        # create models
        print(self.model_conf.name + '_test')
        model = eval(self.model_conf.name + '_test')(self.config)
        # load test model
        print(self.config.save_dir)
        # load_model(model, self.test_conf.test_model)
        load_model(model, self.config.save_dir + self.test_conf.test_model)

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        model.eval()
        correct = 0
        total = 0
        input2innerAll = {}
        for imgs, labels in tqdm(test_loader):
            if self.use_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()

            with torch.no_grad():
                out, _, input2inner = model(imgs, labels)

                for i in range(len(input2inner)):
                    if len(input2innerAll.keys()) < len(input2inner):
                        input2innerAll[i] = input2inner[i]
                    else:
                        input2innerAll[i] = np.vstack((input2innerAll[i], input2inner[i]))

                _, pred = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        test_accuracy = correct / total
        logger.info("Test Accuracy = {} +- {}".format(test_accuracy, 0))
        pickle.dump(input2innerAll, open(os.path.join(self.config.save_dir, 'input2innerAll.p'), 'wb'))

        return test_accuracy



class XorNeuronRunner(object):

    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        self.num_cell_types = config.model.num_cell_types
        self.dataset_conf = config.dataset
        self.model_conf = config.model
        self.pretrain_conf = config.pretrain
        self.train_conf = config.train
        self.test_conf = config.test
        self.use_gpu = config.use_gpu
        self.gpus = config.gpus
        self.writer = SummaryWriter(config.save_dir)

    def pretrain(self, cell_type):
        self.config.seed = int(str(self.seed) + str(cell_type))
        train_dataset = eval(self.dataset_conf.loader_name)(self.config, split='train')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.pretrain_conf.batch_size,
            shuffle=self.pretrain_conf.shuffle,
            num_workers=self.pretrain_conf.num_workers,
            collate_fn=train_dataset.collate_fn,
            drop_last=False)

        # Train innernet
        model = InnerNet(self.config)
        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.pretrain_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.pretrain_conf.lr,
                momentum=self.pretrain_conf.momentum,
                weight_decay=self.pretrain_conf.wd)
        elif self.pretrain_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.pretrain_conf.lr,
                weight_decay=self.pretrain_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        # reset gradient
        optimizer.zero_grad()
        best_train_loss = np.inf
        for epoch in range(self.pretrain_conf.max_epoch):
            train_loss = []
            model.train()
            for xy, targets in train_loader:
                optimizer.zero_grad()

                if self.use_gpu:
                    xy, targets = xy.cuda(), targets.cuda()

                _, loss = model(xy, targets)
                loss.backward()
                optimizer.step()

                # train_loss += [float(loss.data.cpu().numpy())]
                train_loss.append(loss.item())

            # display loss
            train_loss = np.stack(train_loss).mean()

            # save best model
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                print("Pretrain Loss @ epoch {:04d} = {}".format(epoch + 1, np.mean(best_train_loss)))
                snapshot(
                    model.module.inner_net if self.use_gpu else model.inner_net,
                    optimizer,
                    self.config,
                    epoch + 1,
                    tag='best_pretrained' + str(cell_type))

        return 1

    def train_phase1(self):
        # prepare data
        self.config.seed = self.seed
        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            train_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                           train=True,
                                           transform=transform,
                                           download=True)
            train_dataset, val_dataset = random_split(train_dataset, [50000, 10000])
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            train_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                             train=True,
                                             transform=transform,
                                             download=True)
            train_dataset, val_dataset = random_split(train_dataset, [40000, 10000])
        else:
            raise ValueError("Non-supported dataset!")

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.train_conf.batch_size,
                                  shuffle=self.train_conf.shuffle)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.train_conf.batch_size,
                                shuffle=False)

        # create models
        model = eval(self.model_conf.name)(self.config)
        # load pretrained inner-net
        for i in range(self.num_cell_types):
            load_model(model.inner_net[i], self.config.save_dir + self.pretrain_conf.best_model[i])

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum,
                weight_decay=self.train_conf.wd)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_conf.lr_decay_steps,
            gamma=self.train_conf.lr_decay)

        # reset gradient
        optimizer.zero_grad()

        # resume training
        if self.train_conf.is_resume:
            load_model(model, self.train_conf.resume_model, optimizer=optimizer)

        # ========================= Training Loop ============================= #
        iter_count = 0
        results = defaultdict(list)
        best_val_loss = np.inf

        # main loop
        for epoch in range(self.train_conf.max_epoch):
            # ===================== validation ============================ #
            if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
                model.eval()
                val_loss = []
                correct = 0
                total = 0
                for imgs, labels in tqdm(val_loader):
                    if self.use_gpu:
                        imgs, labels = imgs.cuda(), labels.cuda()

                    with torch.no_grad():
                        out, loss, _ = model(imgs, labels)
                    # val_loss += [float(loss.data.cpu().numpy())]
                    val_loss.append(loss.item())

                    _, pred = torch.max(out.data, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

                val_loss = np.stack(val_loss).mean()
                results['val_loss'] += [val_loss]
                results['val_acc'] += [correct / total]
                logger.info("Avg. Validation Loss = {} +- {}".format(val_loss, 0))

                self.writer.add_scalar('val_loss', val_loss, iter_count)

                # save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = correct / total
                    snapshot(
                        model.module.inner_net if self.use_gpu else model.inner_net,
                        optimizer,
                        self.config,
                        epoch + 1,
                        tag='best_phase1')

                logger.info("Current Best Validation Loss = {}".format(best_val_loss))

                from scipy.signal import convolve2d
                from scipy.stats import multivariate_normal
                import matplotlib.pyplot as plt
                nb = 101
                x = np.linspace(-5, 5, nb)
                y = np.linspace(-5, 5, nb)
                xv, yv = np.meshgrid(x, y)
                xy = np.vstack([xv.reshape(-1), yv.reshape(-1)]).T
                mvn = multivariate_normal(mean=[0, 0], cov=[[1/9, 0], [0, 1/9]])
                gaussian_kernel = mvn.pdf(xy).reshape(nb, nb)
                gaussian_kernel /= gaussian_kernel.sum()
                npr = np.random.RandomState(seed=self.config.seed)
                init_unif = npr.uniform(-1, 1, size=(nb, nb))
                targets = convolve2d(init_unif, gaussian_kernel, mode='same').reshape(-1, 1)
                out_phase1 = model.inner_net[0](torch.Tensor(xy))
                plt.imshow(targets.reshape(101, 101), cmap='bwr')
                plt.figure()
                plt.imshow(out_phase1.data.cpu().numpy().reshape(101, 101), cmap='bwr')
                plt.title("{}".format(epoch))
                plt.show()
                import pdb
                pdb.set_trace()

                # check early stop
                if early_stop.tick([val_loss]):
                    snapshot(
                        model.module if self.use_gpu else model,
                        optimizer,
                        self.config,
                        epoch + 1,
                        tag='last')
                    self.writer.close()
                    break
            # ====================== training ============================= #
            model.train()
            lr_scheduler.step()
            for imgs, labels in train_loader:
                # 0. clears all gradients.
                optimizer.zero_grad()
                if self.use_gpu:
                    imgs, labels = imgs.cuda(), labels.cuda()

                # 1. forward pass
                # 2. compute loss
                _, loss, _ = model(imgs, labels)
                # 3. backward pass (accumulates gradients).
                loss.backward()
                # 4. performs a single update step.
                optimizer.step()

                # train_loss = float(loss.data.cpu().numpy())
                train_loss=loss.item()
                results['train_loss'] += [train_loss]
                results['train_step'] += [iter_count]
                self.writer.add_scalar('train_loss', train_loss, iter_count)

                # display loss
                if (iter_count + 1) % self.train_conf.display_iter == 0:
                    logger.info(
                        "Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

                iter_count += 1

            # snapshot model
            if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
                logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
                snapshot(model.module.inner_net if self.use_gpu else model.inner_net, optimizer, self.config, epoch + 1)

        results['best_val_loss'] += [best_val_loss]
        results['best_val_acc'] += [best_val_acc]
        pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats_phase1.p'), 'wb'))
        self.writer.close()
        logger.info("Best Validation Loss = {}".format(best_val_loss))

        return best_val_loss

    def train_phase1_v2(self):
        # 0. 设置种子
        self.config.seed = self.seed
        
        # ========================= 1. 数据集准备 (优化版) ========================= #
        # 定义通用加载参数，开启 pin_memory 和多线程
        loader_kwargs = {
            'batch_size': self.train_conf.batch_size,
            'num_workers': 4,  # 🚀 开启4个子进程加载数据
            'pin_memory': True if self.use_gpu else False,  # 🚀 锁页内存，加速 CPU->GPU 传输
            'persistent_workers': True if self.train_conf.batch_size > 0 else False # 避免每个epoch重新创建线程
        }

        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            full_dataset = datasets.MNIST(root=self.dataset_conf.data_path, train=True, transform=transform, download=True)
            train_dataset, val_dataset = random_split(full_dataset, [50000, 10000])
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
            full_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path, train=True, transform=transform, download=True)
            train_dataset, val_dataset = random_split(full_dataset, [40000, 10000])
        else:
            raise ValueError("Non-supported dataset!")

        train_loader = DataLoader(train_dataset, shuffle=self.train_conf.shuffle, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

        # ========================= 2. 模型与优化器初始化 ========================= #
        model = eval(self.model_conf.name)(self.config)
        
        # 加载预训练 InnerNet
        for i in range(self.num_cell_types):
            # print(model.inner_net[i], self.config.save_dir + self.pretrain_conf.best_model[i])
            load_model(model.inner_net[i], self.config.save_dir + self.pretrain_conf.best_model[i])

        # 设备处理
        device = torch.device('cuda' if self.use_gpu else 'cpu')
        model = model.to(device)
        
        if self.use_gpu and len(self.gpus) > 1:
            model = nn.DataParallel(model, device_ids=self.gpus)

        # 优化器
        params = [p for p in model.parameters() if p.requires_grad]
        opt_conf = self.train_conf
        if opt_conf.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=opt_conf.lr, momentum=opt_conf.momentum, weight_decay=opt_conf.wd)
        elif opt_conf.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=opt_conf.lr, weight_decay=opt_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt_conf.lr_decay_steps, gamma=opt_conf.lr_decay)

        # ========================= 3. 极速训练循环 ========================= #
        iter_count = 0
        results = defaultdict(list)
        best_val_loss = float('inf')
        
        # 预先定义好常量，减少循环内属性查找
        display_iter = opt_conf.display_iter
        snapshot_epoch = opt_conf.snapshot_epoch
        valid_epoch = opt_conf.valid_epoch

        for epoch in range(opt_conf.max_epoch):
            # --------------------- Validation (优化版) --------------------- #
            if (epoch + 1) % valid_epoch == 0 or epoch == 0:
                model.eval()
                val_loss_sum = 0.0
                correct = 0
                total = 0
                
                # 🚀 不计算梯度，不仅省显存，还快
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        if self.use_gpu:
                            imgs = imgs.to(device, non_blocking=True)
                            labels = labels.to(device, non_blocking=True)

                        out, loss, _ = model(imgs, labels)
                        
                        # 🚀 .item() 直接获取数值，不产生 Tensor 开销
                        val_loss_sum += loss.item()
                        
                        _, pred = torch.max(out, 1) # out.data -> out
                        total += labels.size(0)
                        correct += (pred == labels).sum().item()

                # 计算平均值
                avg_val_loss = val_loss_sum / len(val_loader)
                acc = correct / total

                # 记录结果
                results['val_loss'].append(avg_val_loss)
                results['val_acc'].append(acc)
                
                logger.info(f"Avg. Validation Loss = {avg_val_loss:.6f} +- 0")
                self.writer.add_scalar('val_loss', avg_val_loss, iter_count)

                # Save Best
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = acc
                    # 获取原始模型 (处理 DataParallel)
                    model_to_save = model.module.inner_net if hasattr(model, 'module') else model.inner_net
                    snapshot(model_to_save, optimizer, self.config, epoch + 1, tag='best_phase1')
                
                logger.info(f"Current Best Validation Loss = {best_val_loss:.6f}")

                # Early Stop Check
                if early_stop.tick([avg_val_loss]):
                    model_to_save = model.module if hasattr(model, 'module') else model
                    snapshot(model_to_save, optimizer, self.config, epoch + 1, tag='last')
                    self.writer.close()
                    break

            # --------------------- Training (优化版) --------------------- #
            model.train()
            lr_scheduler.step()
            
            for imgs, labels in train_loader:
                # 🚀 set_to_none=True 比 =0 更快
                optimizer.zero_grad(set_to_none=True)
                
                if self.use_gpu:
                    # 🚀 non_blocking=True 实现数据传输与计算流水线并行
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                _, loss, _ = model(imgs, labels)
                loss.backward()
                optimizer.step()

                # 🚀 .item() 提速
                loss_val = loss.item()
                
                results['train_loss'].append(loss_val)
                results['train_step'].append(iter_count)
                self.writer.add_scalar('train_loss', loss_val, iter_count)

                if (iter_count + 1) % display_iter == 0:
                    logger.info(f"Train Loss @ epoch {epoch + 1:04d} iteration {iter_count + 1:08d} = {loss_val:.6f}")

                iter_count += 1

            # Snapshot (Epoch End)
            if (epoch + 1) % snapshot_epoch == 0:
                logger.info(f"Saving Snapshot @ epoch {epoch + 1:04d}")
                model_to_save = model.module if hasattr(model, 'module') else model
                snapshot(model_to_save, optimizer, self.config, epoch + 1)

        # --------------------- Final Wrap up --------------------- #
        results['best_val_loss'].append(best_val_loss)
        results['best_val_acc'].append(best_val_acc) # type: ignore
        pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats_phase1.p'), 'wb'))
        self.writer.close()
        logger.info(f"Best Validation Loss = {best_val_loss:.6f}")

        return best_val_loss
    
    def train_phase2(self):
        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            train_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                           train=True,
                                           transform=transform,
                                           download=True)
            train_dataset, val_dataset = random_split(train_dataset, [50000, 10000])
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            train_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                             train=True,
                                             transform=transform,
                                             download=True)
            train_dataset, val_dataset = random_split(train_dataset, [40000, 10000])
        else:
            raise ValueError("Non-supported dataset!")

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.train_conf.batch_size,
                                  shuffle=self.train_conf.shuffle)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.train_conf.batch_size,
                                shuffle=False)

        # create models
        model = eval(self.model_conf.name)(self.config)
        # load inner-net trained on phase 1
        load_model(model.inner_net, self.config.save_dir + self.train_conf.best_model)

        # ====== Freeze InnerNet  ====== #
        for child in model.children():
            if isinstance(child, nn.ModuleList):
                for ch in child.children():
                    if isinstance(ch, nn.Sequential):  # InnerNet must be the only child built on nn.Sequential()
                        for param in ch.parameters():
                            param.requires_grad = False

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum,
                weight_decay=self.train_conf.wd)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_conf.lr_decay_steps,
            gamma=self.train_conf.lr_decay)

        # reset gradient
        optimizer.zero_grad()

        # resume training
        if self.train_conf.is_resume:
            load_model(model, self.train_conf.resume_model, optimizer=optimizer)

        # ========================= Training Loop ============================= #
        iter_count = 0
        results = defaultdict(list)
        best_val_loss = np.inf
        for epoch in range(self.train_conf.max_epoch):
            # ===================== validation ============================ #
            if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
                model.eval()
                val_loss = []
                correct = 0
                total = 0
                for imgs, labels in tqdm(val_loader):
                    if self.use_gpu:
                        imgs, labels = imgs.cuda(), labels.cuda()

                    with torch.no_grad():
                        out, loss, _ = model(imgs, labels)
                    # val_loss += [float(loss.data.cpu().numpy())]
                    val_loss.append(loss.item())

                    _, pred = torch.max(out.data, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

                val_loss = np.stack(val_loss).mean()
                results['val_loss'] += [val_loss]
                results['val_acc'] += [correct / total]
                logger.info("Avg. Validation Loss = {} +- {}".format(val_loss, 0))

                self.writer.add_scalar('val_loss', val_loss, iter_count)

                # save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = correct / total
                    snapshot(
                        model.module if self.use_gpu else model,
                        optimizer,
                        self.config,
                        epoch + 1,
                        tag='best_phase2')

                logger.info("Current Best Validation Loss = {}".format(best_val_loss))

                # check early stop
                if early_stop.tick([val_loss]):
                    snapshot(
                        model.module if self.use_gpu else model,
                        optimizer,
                        self.config,
                        epoch + 1,
                        tag='last')
                    self.writer.close()
                    break
            # ====================== training ============================= #
            model.train()
            lr_scheduler.step()
            for imgs, labels in train_loader:
                # 0. clears all gradients.
                optimizer.zero_grad()
                if self.use_gpu:
                    imgs, labels = imgs.cuda(), labels.cuda()

                # 1. forward pass
                # 2. compute loss
                _, loss, _ = model(imgs, labels)
                # 3. backward pass (accumulates gradients).
                loss.backward()
                # 4. performs a single update step.
                optimizer.step()

                # train_loss = float(loss.data.cpu().numpy())
                train_loss = loss.item()
                results['train_loss'] += [train_loss]
                results['train_step'] += [iter_count]
                self.writer.add_scalar('train_loss', train_loss, iter_count)

                # display loss
                if (iter_count + 1) % self.train_conf.display_iter == 0:
                    logger.info(
                        "Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

                iter_count += 1

            # snapshot model
            if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
                logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
                snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1)

        results['best_val_loss'] += [best_val_loss]
        results['best_val_acc'] += [best_val_acc]
        pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats_phase2.p'), 'wb'))
        self.writer.close()
        logger.info("Best Validation Loss = {}".format(best_val_loss))

        return best_val_loss

    def test(self):
        print(self.dataset_conf.loader_name)
        print(self.dataset_conf.split)
        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            test_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                          train=False,
                                          transform=transform,
                                          download=True)
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            test_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                            train=False,
                                            transform=transform,
                                            download=True)
        else:
            raise ValueError("Non-supported dataset!")

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.test_conf.batch_size,
                                 shuffle=False)

        # # create models
        # model = eval(self.model_conf.name)(self.config)
        # if 'xor_neuron' in self.model_conf.name:
        #   load_model(model, self.test_conf.test_model)

        # create models
        model = eval(self.model_conf.name)(self.config)
        # load test model
        print(self.config.save_dir)
        # load_model(model, self.test_conf.test_model)
        load_model(model, self.config.save_dir + self.test_conf.test_model)

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        model.eval()
        correct = 0
        total = 0
        in2cells = []
        cnt = 0
        for imgs, labels in tqdm(test_loader):
            if self.use_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()

            with torch.no_grad():
                out, _, in2cells_per_layer_per_batch = model(imgs, labels, collect=True)

                # in2cells : num_layers x [data_size x num_cell_types x ... x arity]
                for i, in2cells_per_layer in enumerate(in2cells_per_layer_per_batch):
                    if cnt == 0:
                        in2cells.append(in2cells_per_layer)
                    elif cnt < 100 and self.dataset_conf.name == 'mnist':
                        in2cells[i] = np.concatenate((in2cells[i], in2cells_per_layer), 0)
                cnt += 1

                _, pred = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        test_accuracy = correct / total
        logger.info("Test Accuracy = {} +- {}".format(test_accuracy, 0))
        pickle.dump(in2cells, open(os.path.join(self.config.save_dir, 'in2cells.p'), 'wb'))

        return test_accuracy

    def test_local(self):
        print(self.dataset_conf.loader_name)
        print(self.dataset_conf.split)
        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            test_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                          train=False,
                                          transform=transform,
                                          download=True)
        elif self.dataset_conf.name == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            test_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                            train=False,
                                            transform=transform,
                                            download=True)
        else:
            raise ValueError("Non-supported dataset!")

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.test_conf.batch_size,
                                 shuffle=False)

        # # create models
        # model = eval(self.model_conf.name)(self.config)
        # if 'xor_neuron' in self.model_conf.name:
        #   load_model(model, self.test_conf.test_model)

        # create models
        model = eval(self.model_conf.name + '_test')(self.config)
        # load test model
        print(self.config.save_dir)
        load_model(model, self.test_conf.test_model)

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        model.eval()
        correct = 0
        total = 0
        input2innerAll = {}
        for imgs, labels in tqdm(test_loader):
            if self.use_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()

            with torch.no_grad():
                out, _, input2inner = model(imgs, labels)

                for i in range(len(input2inner)):
                    if len(input2innerAll.keys()) < len(input2inner):
                        input2innerAll[i] = input2inner[i]
                    else:
                        input2innerAll[i] = np.vstack((input2innerAll[i], input2inner[i]))

                _, pred = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        test_accuracy = correct / total
        logger.info("Test Accuracy = {} +- {}".format(test_accuracy, 0))
        pickle.dump(input2innerAll, open(os.path.join(self.config.save_dir, 'input2innerAll.p'), 'wb'))

        return test_accuracy


class XorNeuronLMRunner(object):

    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        self.num_cell_types = config.model.num_cell_types
        self.dataset_conf = config.dataset
        self.model_conf = config.model
        self.pretrain_conf = config.pretrain
        self.train_conf = config.train
        self.test_conf = config.test
        self.use_gpu = config.use_gpu
        self.gpus = config.gpus
        self.writer = SummaryWriter(config.save_dir)
        self.corpus = Corpus(self.dataset_conf.data_path + '/ptb')
        self.ntokens = len(self.corpus.dictionary)  # 10000

    def pretrain(self, cell_type):
        self.config.seed = int(str(self.seed) + str(cell_type))
        train_dataset = eval(self.dataset_conf.loader_name)(self.config, split='train')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.pretrain_conf.batch_size,
            shuffle=self.pretrain_conf.shuffle,
            num_workers=self.pretrain_conf.num_workers,
            collate_fn=train_dataset.collate_fn,
            drop_last=False)

        # Train innernet
        model = InnerNet(self.config)
        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.pretrain_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.pretrain_conf.lr,
                momentum=self.pretrain_conf.momentum,
                weight_decay=self.pretrain_conf.wd)
        elif self.pretrain_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.pretrain_conf.lr,
                weight_decay=self.pretrain_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        # reset gradient
        optimizer.zero_grad()
        best_train_loss = np.inf
        for epoch in range(self.pretrain_conf.max_epoch):
            train_loss = []
            model.train()
            for xy, targets in train_loader:
                optimizer.zero_grad()

                if self.use_gpu:
                    xy, targets = xy.cuda(), targets.cuda()

                _, loss = model(xy, targets)
                loss.backward()
                optimizer.step()

                train_loss += [float(loss.data.cpu().numpy())]

            # display loss
            train_loss = np.stack(train_loss).mean()

            # save best model
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                print("Pretrain Loss @ epoch {:04d} = {}".format(epoch + 1, np.mean(best_train_loss)))
                snapshot(
                    model.module.inner_net if self.use_gpu else model.inner_net,
                    optimizer,
                    self.config,
                    epoch + 1,
                    tag='best_pretrained' + str(cell_type))

        return 1

    def train_phase1(self):
        self.config.seed = self.seed
        if self.dataset_conf.name == 'ptb':
            eval_batch_size = 10
            # data : seq_len x batch_size
            train_data = self.batchify(self.corpus.train, self.train_conf.batch_size)
            val_data = self.batchify(self.corpus.valid, self.train_conf.batch_size)
        else:
            raise ValueError("Non-supported dataset!")

        # create models
        model = eval(self.model_conf.name)(self.config, self.ntokens)
        # load pretrained inner-net
        for i in range(self.num_cell_types):
            load_model(model.inner_net[i], self.config.save_dir + self.pretrain_conf.best_model[i])

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum,
                weight_decay=self.train_conf.wd)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_conf.lr_decay_steps,
            gamma=self.train_conf.lr_decay)

        # reset gradient
        optimizer.zero_grad()

        # resume training
        if self.train_conf.is_resume:
            load_model(model, self.train_conf.resume_model, optimizer=optimizer)

        # ========================= Training Loop ============================= #
        iter_count = 0
        results = defaultdict(list)
        best_val_loss = np.inf
        for epoch in range(self.train_conf.max_epoch):
            # ===================== validation ============================ #
            if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
                model.eval()
                val_loss = []
                hidden = None
                for i in range(0, val_data.size(0) - 1, self.train_conf.bptt):  # iterate over every timestep
                    data, targets = self.get_batch(val_data, i)
                    if self.use_gpu:
                        data, targets = data.cuda(), targets.cuda()
                    with torch.no_grad():
                        _, hidden, loss, _ = model(data, targets, mask=None, hx=hidden)

                    hidden = self.repackage_hidden(hidden)
                    val_loss += [float(loss.data.cpu().numpy())]

            val_loss = np.stack(val_loss).mean()
            results['val_loss'] += [val_loss]
            logger.info("Avg. Validation Loss = {} +- {}".format(val_loss, 0))

            self.writer.add_scalar('val_loss', val_loss, iter_count)

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                snapshot(
                    model.module.inner_net if self.use_gpu else model.inner_net,
                    optimizer,
                    self.config,
                    epoch + 1,
                    tag='best_phase1')

            logger.info("Current Best Validation Loss = {}".format(best_val_loss))

            # check early stop
            if early_stop.tick([val_loss]):
                snapshot(
                    model.module if self.use_gpu else model,
                    optimizer,
                    self.config,
                    epoch + 1,
                    tag='last')
                self.writer.close()
                break
        # ====================== training ============================= #

            model.train()
            lr_scheduler.step()
            hidden = None
            mask = [None] * len(self.model_conf.out_hidden_dim)
            for i in range(len(self.model_conf.out_hidden_dim)):
                mask[i] = torch.bernoulli(torch.Tensor(self.model_conf.out_hidden_dim[i]).fill_(1 - self.model_conf.dropout)) / (1 - self.model_conf.dropout)

            for batch, i in enumerate(range(0, train_data.size(0) - 1, self.train_conf.bptt)):
                # 0. clears all gradients.
                optimizer.zero_grad()
                data, targets = self.get_batch(train_data, i)
                if self.use_gpu:
                    data, targets = data.cuda(), targets.cuda()
                _, hidden, loss, _ = model(data, targets, mask=mask, hx=hidden)
                hidden = self.repackage_hidden(hidden)
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(model.parameters(), self.train_conf.clip)
                optimizer.step()

                train_loss = float(loss.data.cpu().numpy())
                results['train_loss'] += [train_loss]
                results['train_step'] += [iter_count]
                self.writer.add_scalar('train_loss', train_loss, iter_count)

                # display loss
                if (iter_count + 1) % self.train_conf.display_iter == 0:
                    logger.info(
                        "Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

                iter_count += 1

            # snapshot model
            if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
                logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
                snapshot(model.module.inner_net if self.use_gpu else model.inner_net, optimizer, self.config, epoch + 1)

        results['best_val_loss'] += [best_val_loss]
        pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats_phase1.p'), 'wb'))
        self.writer.close()
        logger.info("Best Validation Loss = {}".format(best_val_loss))

        return best_val_loss

    def train_phase2(self):
        if self.dataset_conf.name == 'ptb':
            eval_batch_size = 10
            # data : seq_len x batch_size
            train_data = self.batchify(self.corpus.train, self.train_conf.batch_size)
            val_data = self.batchify(self.corpus.valid, self.train_conf.batch_size)
        else:
            raise ValueError("Non-supported dataset!")

        # create models
        model = eval(self.model_conf.name)(self.config, self.ntokens)
        # load inner-net trained on phase 1
        load_model(model.inner_net, self.config.save_dir + self.train_conf.best_model)

        # ====== Freeze InnerNet  ====== #
        for child in model.children():
            if isinstance(child, nn.ModuleList):
                for ch in child.children():
                    if isinstance(ch, nn.Sequential):  # InnerNet must be the only child built on nn.Sequential()
                        for param in ch.parameters():
                            param.requires_grad = False

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum,
                weight_decay=self.train_conf.wd)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_conf.lr_decay_steps,
            gamma=self.train_conf.lr_decay)

        # reset gradient
        optimizer.zero_grad()

        # resume training
        if self.train_conf.is_resume:
            load_model(model, self.train_conf.resume_model, optimizer=optimizer)

        # ========================= Training Loop ============================= #
        iter_count = 0
        results = defaultdict(list)
        best_val_loss = np.inf
        for epoch in range(self.train_conf.max_epoch):
            # ===================== validation ============================ #
            if (epoch + 1) % self.train_conf.valid_epoch == 0 or epoch == 0:
                model.eval()
                val_loss = []
                hidden = None
                for i in range(0, val_data.size(0) - 1, self.train_conf.bptt):  # iterate over every timestep
                    data, targets = self.get_batch(val_data, i)
                    if self.use_gpu:
                        data, targets = data.cuda(), targets.cuda()
                    with torch.no_grad():
                        _, hidden, loss, _ = model(data, targets, mask=None, hx=hidden)

                    hidden = self.repackage_hidden(hidden)
                    val_loss += [float(loss.data.cpu().numpy())]

            val_loss = np.stack(val_loss).mean()
            results['val_loss'] += [val_loss]
            logger.info("Avg. Validation Loss = {} +- {}".format(val_loss, 0))

            self.writer.add_scalar('val_loss', val_loss, iter_count)

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                snapshot(
                    model.module if self.use_gpu else model,
                    optimizer,
                    self.config,
                    epoch + 1,
                    tag='best_phase2')

            logger.info("Current Best Validation Loss = {}".format(best_val_loss))

            # check early stop
            if early_stop.tick([val_loss]):
                snapshot(
                    model.module if self.use_gpu else model,
                    optimizer,
                    self.config,
                    epoch + 1,
                    tag='last')
                self.writer.close()
                break
        # ====================== training ============================= #

            model.train()
            lr_scheduler.step()
            hidden = None
            mask = [None] * len(self.model_conf.out_hidden_dim)
            for i in range(len(self.model_conf.out_hidden_dim)):
                mask[i] = torch.bernoulli(torch.Tensor(self.model_conf.out_hidden_dim[i]).fill_(1 - self.model_conf.dropout)) / (1 - self.model_conf.dropout)

            for batch, i in enumerate(range(0, train_data.size(0) - 1, self.train_conf.bptt)):
                # 0. clears all gradients.
                optimizer.zero_grad()
                data, targets = self.get_batch(train_data, i)
                if self.use_gpu:
                    data, targets = data.cuda(), targets.cuda()
                _, hidden, loss, _ = model(data, targets, mask=mask, hx=hidden)
                hidden = self.repackage_hidden(hidden)
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(model.parameters(), self.train_conf.clip)
                optimizer.step()

                train_loss = float(loss.data.cpu().numpy())
                results['train_loss'] += [train_loss]
                results['train_step'] += [iter_count]
                self.writer.add_scalar('train_loss', train_loss, iter_count)

                # display loss
                if (iter_count + 1) % self.train_conf.display_iter == 0:
                    logger.info(
                        "Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

                iter_count += 1

            # snapshot model
            if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
                logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
                snapshot(model.module.inner_net if self.use_gpu else model.inner_net, optimizer, self.config, epoch + 1)

        results['best_val_loss'] += [best_val_loss]
        pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats_phase2.p'), 'wb'))
        self.writer.close()
        logger.info("Best Validation Loss = {}".format(best_val_loss))

        return best_val_loss

    def test(self):
        print(self.dataset_conf.loader_name)
        print(self.dataset_conf.split)
        if self.dataset_conf.name == 'ptb':
            eval_batch_size = 10
            # data : seq_len x batch_size
            test_data = self.batchify(self.corpus.test, self.test_conf.batch_size)
        else:
            raise ValueError("Non-supported dataset!")

        # create models
        model = eval(self.model_conf.name)(self.config, self.ntokens)
        # load test model
        print(self.config.save_dir)
        # load_model(model, self.test_conf.test_model)
        load_model(model, self.config.save_dir + self.test_conf.test_model)

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        model.eval()
        in2cells = []
        cnt = 0
        test_loss = []
        hidden = None
        for i in range(0, test_data.size(0) - 1, self.train_conf.bptt):
            data, targets = self.get_batch(test_data, i)
            if self.use_gpu:
                data, targets = data.cuda(), targets.cuda()
            with torch.no_grad():
                _, hidden, loss, in2cells_per_layer_per_batch = model(data, targets, mask=None, hx=hidden)

                # in2cells : num_layers x [data_size x num_cell_types x ... x arity]
                for i, in2cells_per_layer in enumerate(in2cells_per_layer_per_batch):
                    if cnt == 0:
                        in2cells.append(in2cells_per_layer)
                    else:
                        in2cells[i] = np.concatenate((in2cells[i], in2cells_per_layer), 0)
                cnt += 1

                hidden = self.repackage_hidden(hidden)
                test_loss += [float(loss.data.cpu().numpy())]

        test_loss = np.stack(test_loss).mean()
        logger.info("Avg. Test Loss = {} +- {}".format(test_loss, 0))
        pickle.dump(in2cells, open(os.path.join(self.config.save_dir, 'in2cells.p'), 'wb'))

        return test_loss

    def batchify(self, data, batch_size):
        seq_len = data.size(0) // batch_size
        data = data.narrow(0, 0, seq_len * batch_size)
        # data : seq_len, batch_size
        data = data.view(batch_size, -1).t().contiguous()
        return data

    def repackage_hidden(self, h):
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return [self.repackage_hidden(v) for v in h]

    def get_batch(self, source, i):
        seq_len = min(self.train_conf.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target
