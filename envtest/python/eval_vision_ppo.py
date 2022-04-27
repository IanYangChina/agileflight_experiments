#!/usr/bin/env python3
import argparse
import math
#
import os
import time
import numpy as np
import torch
from flightgym import VisionEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy

from agileflight_policy.envs import vec_env_wrapper as wrapper
from agileflight_policy.common.util import test_policy


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--render", type=int, default=1, help="Render with Unity")
    parser.add_argument("--trial", type=int, default=1, help="PPO trial number")
    parser.add_argument("--iter", type=int, default=2000, help="PPO iter number")
    parser.add_argument("--logdir", type=str, default="", help="Case log folder name")
    return parser


def main():
    args = parser().parse_args()

    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + '/' + args.logdir
    tb_log_name = 'PPO'
    os.makedirs(log_dir, exist_ok=True)

    # load configurations
    cfg = YAML().load(
        open(
            rsg_root + "/env_config.yaml", "r"
        )
    )

    # create evaluation environment
    if args.render:
        cfg["unity"]["render"] = "yes"
    cfg["simulation"]["num_envs"] = 1
    cfg["environment"]["level"] = "medium"
    eval_env = wrapper.FlightEnvVec(
        VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    )

    #
    os.system(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64 &")
    #
    weight = log_dir + "/" + tb_log_name + "_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
    env_rms = log_dir + "/" + tb_log_name + "_{0}/RMS/iter_{1:05d}.npz".format(args.trial, args.iter)

    device = get_device("auto")
    saved_variables = torch.load(weight, map_location=device)
    # Create policy object
    policy = MlpPolicy(**saved_variables["data"])
    #
    policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
    # Load weights
    policy.load_state_dict(saved_variables["state_dict"], strict=False)
    policy.to(device)
    #
    eval_env.load_rms(env_rms)
    test_policy(eval_env, policy, render=args.render)


if __name__ == "__main__":
    start = time.time()
    main()
    passed_time = ((time.time() - start) / 60) / 60
    print("Process running time: %f hours" % passed_time)
