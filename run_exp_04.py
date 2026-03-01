import os
import sys
import torch
import logging
import traceback
import numpy as np
from pprint import pprint

from runner import *
from utils.logger import setup_logging
from utils.arg_helper import parse_arguments, get_config
torch.set_printoptions(profile='full')

#===============================================================
import datajoint as dj
schema = dj.schema('kijung-gnn-04')

@schema
class Sample(dj.Manual):
  definition = """
    sample_id : int unsigned
    """
@schema
class SampleComputed(dj.Computed):
  definition = """
    -> Sample
    """
  def make(self, key):
    sample_id = key['sample_id']
    # Call main function to start computation
    main(sample_id)
    self.insert1(key)  # Just to mark it as completed
#===============================================================

# python3 run_exp.py -c config/bp.yaml -t
def main(sample_id):
  args = parse_arguments()
  config = get_config(args.config_file, sample_id="{:03d}".format(sample_id))
  # args.config_file = args.config_file[:-5] + "_{:03d}.yaml".format(sample_id)
  # config = get_config(args.config_file, sample_id="{:03d}".format(sample_id))
  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
  torch.cuda.manual_seed_all(config.seed)
  config.use_gpu = config.use_gpu and torch.cuda.is_available()

  # log info
  log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.run_id))
  logger = setup_logging(args.log_level, log_file)
  logger.info("Writing log file to {}".format(log_file))
  logger.info("Exp instance id = {}".format(config.run_id))
  logger.info("Exp comment = {}".format(args.comment))
  logger.info("Config =")
  print(">" * 80)
  pprint(config)
  print("<" * 80)

  # Run the experiment
  try:
    runner = eval(config.runner)(config)
    if not args.test:
      runner.pretrain(1)
      runner.pretrain(2)
      runner.pretrain(3)
      runner.pretrain(4)
      runner.pretrain(5)
      runner.pretrain(6)
      runner.pretrain(7)
      runner.pretrain(8)
      runner.pretrain(9)
      runner.pretrain(10)

      runner.train_phase1()
      runner.train_phase2()
      runner.test()
    else:
      runner.test_local()
  except:
    logger.error(traceback.format_exc())

  sys.exit(0)


if __name__ == "__main__":
  # main()
  SampleComputed().populate(reserve_jobs=True, suppress_errors=False)