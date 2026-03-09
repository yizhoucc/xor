"""Unified experiment entry point with config hash-based deduplication and checkpoint resumption.

Usage:
    python run.py -c config/experiments/mlp_mnist_2arg.yaml
    python run.py -c config/experiments/mlp_mnist_2arg.yaml -t          # test only
    python run.py -c config/experiments/mlp_mnist_2arg.yaml --resume exp/xxx  # manual resume
"""
import os
import sys
import copy
import hashlib
import argparse
import time
import yaml
import numpy as np
import torch

from easydict import EasyDict as edict
from utils.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="XOR Neuron Experiments")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument('-t', '--test', action='store_true',
                        help="Test only (skip training)")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to existing experiment dir to resume")
    parser.add_argument('--seed', type=int, default=None,
                        help="Override seed from config (for multi-seed experiments)")
    return parser.parse_args()


def compute_config_hash(config_dict):
    """Compute SHA256 hash of config, excluding path/run-specific fields."""
    d = copy.deepcopy(config_dict)
    for key in ['exp_dir', 'save_dir', 'run_id']:
        d.pop(key, None)
    # reuse_from is a runtime shortcut, not a model/training parameter
    if 'pretrain' in d and isinstance(d['pretrain'], dict):
        d['pretrain'].pop('reuse_from', None)
    serialized = yaml.dump(d, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def find_existing_experiment(exp_base_dir, config_hash):
    """Scan exp/ for a matching config hash. Returns (save_dir, is_completed) or (None, False)."""
    if not os.path.isdir(exp_base_dir):
        return None, False

    for entry in os.listdir(exp_base_dir):
        entry_path = os.path.join(exp_base_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        hash_file = os.path.join(entry_path, 'config_hash.txt')
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            if stored_hash == config_hash:
                is_completed = os.path.exists(os.path.join(entry_path, 'COMPLETED'))
                return entry_path, is_completed

    return None, False


def get_completed_stage(save_dir):
    """Determine which stages have been completed based on marker files."""
    stages = ['PRETRAIN_DONE', 'PHASE1_DONE', 'PHASE2_DONE', 'TEST_DONE']
    completed = []
    for stage in stages:
        if os.path.exists(os.path.join(save_dir, stage)):
            completed.append(stage)
    return completed


def edict2dict(edict_obj):
    """Convert EasyDict to regular dict recursively."""
    result = {}
    for key, val in edict_obj.items():
        if isinstance(val, edict):
            result[key] = edict2dict(val)
        else:
            result[key] = val
    return result


def main():
    args = parse_args()

    # 1. Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # Override seed if specified via CLI
    if args.seed is not None:
        config_dict['seed'] = args.seed

    config = edict(config_dict)
    exp_base_dir = config.get('exp_dir', 'exp')

    # 2. Compute config hash (includes seed, so different seeds get different dirs)
    config_hash = compute_config_hash(config_dict)

    # 3. Determine save directory
    if args.resume:
        save_dir = args.resume
        print(f"Resuming from: {save_dir}")
    else:
        existing_dir, is_completed = find_existing_experiment(exp_base_dir, config_hash)
        if is_completed:
            print(f"Found completed experiment: {existing_dir}")
            print("Nothing to do. Exiting.")
            return
        elif existing_dir:
            save_dir = existing_dir
            print(f"Found incomplete experiment, resuming: {save_dir}")
        else:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            exp_name = config.get('exp_name', 'experiment')
            dir_name = f"{exp_name}_{timestamp}_{config_hash[:8]}"
            save_dir = os.path.join(exp_base_dir, dir_name)
            os.makedirs(save_dir, exist_ok=True)
            print(f"Starting new experiment: {save_dir}")

    config.save_dir = save_dir
    config.run_id = str(os.getpid())
    config.use_gpu = config.get('use_gpu', True) and torch.cuda.is_available()

    # 4. Save config and hash
    os.makedirs(save_dir, exist_ok=True)

    config_save_path = os.path.join(save_dir, 'config.yaml')
    if not os.path.exists(config_save_path):
        yaml.dump(edict2dict(config), open(config_save_path, 'w'), default_flow_style=False)

    hash_save_path = os.path.join(save_dir, 'config_hash.txt')
    with open(hash_save_path, 'w') as f:
        f.write(config_hash)

    # 5. Set seeds
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 6. Setup logging
    log_file = os.path.join(save_dir, f"log_exp_{config.run_id}.txt")
    logger = setup_logging('INFO', log_file)
    logger.info(f"Config: {args.config}")
    logger.info(f"Config hash: {config_hash}")
    logger.info(f"Save dir: {save_dir}")

    # 7. Run experiment
    task_type = config.get('task_type', 'classification')
    logger.info(f"Task type: {task_type}")

    if task_type == 'rl':
        from runner.rl_runner import RLRunner
        runner = RLRunner(config)
        if args.test:
            runner.test()
        else:
            runner.train()
    elif task_type == 'language_model':
        from runner.lm_runner import LMRunner
        runner = LMRunner(config)
        if args.test:
            runner.test()
        else:
            runner.train()
    else:
        from runner.experiment_runner import ExperimentRunner
        runner = ExperimentRunner(config)

        if args.test:
            runner.test()
            return

        completed_stages = get_completed_stage(save_dir)

        if 'PRETRAIN_DONE' not in completed_stages:
            logger.info("=== Starting Pretrain ===")
            runner.pretrain()
        else:
            logger.info("Pretrain already done, skipping.")

        if 'PHASE1_DONE' not in completed_stages:
            logger.info("=== Starting Phase 1 ===")
            runner.train_phase1()
        else:
            logger.info("Phase 1 already done, skipping.")

        if 'PHASE2_DONE' not in completed_stages:
            logger.info("=== Starting Phase 2 ===")
            runner.train_phase2()
        else:
            logger.info("Phase 2 already done, skipping.")

        if 'TEST_DONE' not in completed_stages:
            logger.info("=== Starting Test ===")
            runner.test()
        else:
            logger.info("Test already done, skipping.")

    logger.info("Experiment completed!")


if __name__ == "__main__":
    main()
