import os
from os.path import join as ospj
import sys
import time
import shutil
from datetime import datetime, timedelta
import random
import numpy as np
import torch
import torch.nn as nn
import imageio

from stl_lib import softmax_pairs, softmin_pairs, softmax, softmin

def build_relu_nn(input_dim, output_dim, hiddens, activation_fn, last_fn=None):
    n_neurons = [input_dim] + hiddens + [output_dim]
    layers = []
    for i in range(len(n_neurons)-1):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
        layers.append(activation_fn())
    if last_fn is not None:
        layers[-1] = last_fn()
    else:
        del layers[-1]
    return nn.Sequential(*layers)

def to_np(x):
    return x.detach().cpu().numpy()

def to_torch(x):
    return torch.from_numpy(x).float().cuda()

def uniform_tensor(amin, amax, size):
    return torch.rand(size) * (amax - amin) + amin

def rand_choice_tensor(choices, size):
    return torch.from_numpy(np.random.choice(choices, size)).float()


def generate_gif(gif_path, duration, fs_list):
    with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
        for filename in fs_list:
            image = imageio.imread(filename)
            writer.append_data(image)


def soft_step(x):
    return (torch.tanh(500 * x) + 1)/2

# TODO get the exp dir
def get_exp_dir(just_local=False):
    if just_local:
        return "./"
    else:
        for poss_dir in ["../exps_stl/", "../../exps_stl/", "/datadrive/exps_stl/"]:
            if os.path.exists(poss_dir):
                return poss_dir
    exit("no available exp directory! Exit...")


def find_path(path):
    for poss_dir in ["./", "../exps_stl/", "../../exps_stl/", "/datadrive/exps_stl/"]:
        the_path = os.path.join(poss_dir,path)
        if os.path.exists(the_path):
            return the_path

class EtaEstimator():
    def __init__(self, start_iter, end_iter, check_freq, num_workers=1):
        self.start_iter = start_iter
        num_workers = 1 if num_workers is None else num_workers
        self.end_iter = end_iter//num_workers
        self.check_freq = check_freq
        self.curr_iter = start_iter
        self.start_timer = None
        self.interval = 0
        self.eta_t = 0
        self.num_workers = num_workers

    def update(self):
        if self.start_timer is None:
            self.start_timer = time.time()
        self.curr_iter += 1
        if self.curr_iter % (max(1,self.check_freq//self.num_workers)) == 0:
            self.interval = self.elapsed() / (self.curr_iter - self.start_iter)        
            self.eta_t = self.interval * (self.end_iter - self.curr_iter)
    
    def elapsed(self):
        return time.time() - self.start_timer
    
    def eta(self):
        return self.eta_t
    
    def elapsed_str(self):
        return time_format(self.elapsed())
    
    def interval_str(self):
        return time_format(self.interval)

    def eta_str(self):
        return time_format(self.eta_t)

def time_format(secs):
    _s = secs % 60 
    _m = secs % 3600 // 60
    _h = secs % 86400 // 3600
    _d = secs // 86400
    if _d != 0:
        return "%02dD%02dh%02dm%02ds"%(_d, _h, _m, _s)
    else:
        if _h != 0:
            return "%02dH%02dm%02ds"%(_h, _m, _s)
        else:
            if _m != 0:
                return "%02dm%02ds"%(_m, _s)
            else:
                return "%05.2fs"%(_s)


# TODO create the exp directory
def setup_exp_and_logger(args, set_gpus=True, just_local=False, test=False, ford=False, ford_debug=False):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sys.stdout = logger = Logger()
    EXP_ROOT_DIR = get_exp_dir(just_local)

    if test:
        if (hasattr(args, "rl") and args.rl) or (hasattr(args, "mbpo") and args.mbpo) or (hasattr(args, "pets") and args.pets):
            tuples = args.rl_path.split("/")
        else:
            tuples = args.net_pretrained_path.split("/")
        if ".ckpt" in tuples[-1] or ".zip" in tuples[-1]:
            EXP_ROOT_DIR = ospj(EXP_ROOT_DIR, tuples[-3])
        elif ".pth" in tuples[-1]:  # mbrl case
            EXP_ROOT_DIR = ospj(EXP_ROOT_DIR, tuples[-2])
        else:
            EXP_ROOT_DIR = ospj(EXP_ROOT_DIR, tuples[-1])
        if ford:
            args.exp_dir_full = os.path.join(EXP_ROOT_DIR, "test_ford_%s" % (logger._timestr))
        elif ford_debug:
            args.exp_dir_full = os.path.join(EXP_ROOT_DIR, "test_ford_debug_%s" % (logger._timestr))
        else:
            args.exp_dir_full = os.path.join(EXP_ROOT_DIR, "test_%s" % (logger._timestr))
    else:
        if args.exp_name.startswith("exp") and "debug" not in str.lower(args.exp_name) and "dbg" not in str.lower(args.exp_name):
            args.exp_dir_full = os.path.join(EXP_ROOT_DIR, args.exp_name)
        else:
            args.exp_dir_full = os.path.join(EXP_ROOT_DIR, "g%s_%s" % (logger._timestr, args.exp_name))
    args.viz_dir = os.path.join(args.exp_dir_full, "viz")
    args.src_dir = os.path.join(args.exp_dir_full, "src")
    args.model_dir = os.path.join(args.exp_dir_full, "models")
    os.makedirs(args.viz_dir, exist_ok=True)
    os.makedirs(args.src_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    for fname in os.listdir('./'):
        if fname.endswith('.py'):
            shutil.copy(fname, os.path.join(args.src_dir, fname))

    logger.create_log(args.exp_dir_full)
    write_cmd_to_file(args.exp_dir_full, sys.argv)
    np.savez(os.path.join(args.exp_dir_full, 'args'), args=args)

    if set_gpus and hasattr(args, "gpus") and args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    return args


# TODO logger
class Logger(object):
    def __init__(self):
        self._terminal = sys.stdout
        self._timestr = datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")
        self.log = None

    def create_log(self, log_path):
        self.log = open(log_path + "/log-%s.txt" % self._timestr, "a", 1)

    def write(self, message):
        self._terminal.write(message)
        if self.log is not None:
            self.log.write(message)

    def flush(self):
        pass


def write_cmd_to_file(log_dir, argv):
    with open(ospj(log_dir, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(argv))
