import os
import torch


def data_to_gpu(*input_data):
  return_data = []
  for dd in input_data:
    if type(dd).__name__ == 'Tensor':
      return_data += [dd.cuda()]
  
  return tuple(return_data)


def snapshot(model, optimizer, config, step, gpus=[0], tag=None):
  model_snapshot = {
      "model": model.state_dict(),
      "optimizer": optimizer.state_dict(),
      "step": step
  }

  torch.save(model_snapshot,
             os.path.join(config.save_dir, "model_snapshot_{}.pth".format(tag)
                          if tag is not None else
                          "model_snapshot_{:07d}.pth".format(step)))


def load_model_old(model, file_name, optimizer=None):
  model_snapshot = torch.load(file_name, map_location=torch.device('cpu'))
  model.load_state_dict(model_snapshot["model"], strict=True)

  if optimizer is not None:
    optimizer.load_state_dict(model_snapshot["optimizer"])

def load_model(model, file_name, optimizer=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_snapshot = torch.load(file_name, map_location=device)

    # Strip '_orig_mod.' prefix from keys saved by torch.compile
    state_dict = model_snapshot["model"]
    if any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(model_snapshot["optimizer"])
    print(f"Loaded model from {file_name} to {device}")

def load_model_v3(model_v3, best_model_paths, save_dir):
    """
    专门为 V3 架构设计的加载函数
    将多个独立的预训练 InnerNet (Linear层) 权重合并映射到 V3 的 Grouped Conv1d 权重中
    """
    import torch
    
    # 1. 依次读取所有 cell 类型的预训练权重文件
    all_state_dicts = []
    for path in best_model_paths:
        full_path = save_dir + path
        all_state_dicts.append(torch.load(full_path, map_location='cpu'))

    # 2. 遍历 InnerNet 的 3 层微型结构 (fc1, fc2, fc3)
    with torch.no_grad():
        for i, old_layer_name in enumerate(['fc1', 'fc2', 'fc3']):
            # 提取所有 cell 在这一层的 weight [Out, In] 和 bias [Out]
            weights = [sd['model'][f'{old_layer_name}.weight'] for sd in all_state_dicts]
            biases = [sd['model'][f'{old_layer_name}.bias'] for sd in all_state_dicts]
            
            # 🚀 权重映射核心：
            # V3 的 Conv1d 权重期望形状为 [Out_total, In_per_group, 1]
            # 我们将多个 [Out, In] 的 Linear 权重在第 0 维（输出维）拼接，并增加一个宽度为 1 的卷积维度
            combined_weight = torch.cat(weights, dim=0).unsqueeze(-1)
            combined_bias = torch.cat(biases, dim=0)
            
            # 写入 V3 模型的 Conv1d 层 (层名对应 model_v3.inner_net[i])
            model_v3.inner_net[i].weight.copy_(combined_weight)
            model_v3.inner_net[i].bias.copy_(combined_bias)
            
    print(f"✅ Successfully mapped {len(all_state_dicts)} pretrained InnerNets to V3 architecture.")
    
class EarlyStopper(object):
  """ 
    Check whether the early stop condition (always 
    observing decrease in a window of time steps) is met.

    Usage:
      my_stopper = EarlyStopper([0, 0], 1)
      is_stop = my_stopper.tick([-1,-1]) # returns True
  """

  def __init__(self, init_val, win_size=10, is_decrease=True):
    if not isinstance(init_val, list):
      raise ValueError("EarlyStopper only takes list of int/floats")

    self._win_size = win_size
    self._num_val = len(init_val)
    self._val = [[False] * win_size for _ in range(self._num_val)]
    self._last_val = init_val[:]
    self._comp_func = (lambda x, y: x < y) if is_decrease else (lambda x, y: x >= y)

  def tick(self, val):
    if not isinstance(val, list):
      raise ValueError("EarlyStopper only takes list of int/floats")

    assert len(val) == self._num_val

    for ii in range(self._num_val):
      self._val[ii].pop(0)

      if self._comp_func(val[ii], self._last_val[ii]):
        self._val[ii].append(True)
      else:
        self._val[ii].append(False)

      self._last_val[ii] = val[ii]

    is_stop = all([all(xx) for xx in self._val])

    return is_stop
