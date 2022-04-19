#!/usr/bin/env python3
import argparse
#
import os
import time
import numpy as np
import torch
from flightgym import VisionEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump

from agileflight_policy.common.ppo import PPO
from agileflight_policy.envs import vec_env_wrapper as wrapper


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--lvcoeff", type=float, default=-0.01, help="Env linear velocity penalty coefficient")
    parser.add_argument("--crcoeff", type=float, default=-0.01, help="Env collision reward coefficient")
    parser.add_argument("--avcoeff", type=float, default=-0.0001, help="Env angular velocity penalty coefficient")
    return parser


def main():
    args = parser().parse_args()

    # load configurations
    cfg = YAML().load(
        open(
            os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
        )
    )

    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + \
              '/crcoeff_' + str(args.crcoeff) + \
              '_lvcoeff_' + str(args.lvcoeff) + \
              '_avcoeff_' + str(args.avcoeff)
    tb_log_name = 'PPO'
    os.makedirs(log_dir, exist_ok=True)

    # create training environment
    cfg["rewards"]["vel_coeff"] = float(args.lvcoeff)
    cfg["rewards"]["collision_coeff"] = float(args.crcoeff)
    cfg["rewards"]["angular_vel_coeff"] = float(args.avcoeff)
    cfg["simulation"]["num_threads"] = 5
    train_env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)

    # set random seed
    configure_random_seed(args.seed, env=train_env)

    cfg["unity"]["render"] = "no"

    # create evaluation environment
    old_num_envs = cfg["simulation"]["num_envs"]
    cfg["simulation"]["num_envs"] = 1
    eval_env = wrapper.FlightEnvVec(
        VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    )
    cfg["simulation"]["num_envs"] = old_num_envs

    #
    model = PPO(
        tensorboard_log=log_dir,
        policy="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[dict(pi=[256, 256], vf=[512, 512])],
            log_std_init=-0.5,
        ),
        env=train_env,
        eval_env=eval_env,
        use_tanh_act=True,
        learning_rate=3e-4,
        gae_lambda=0.95,
        gamma=0.99,
        n_steps=250,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        batch_size=25000,
        clip_range=0.2,
        use_sde=False,  # don't use (gSDE), doesn't work
        env_cfg=cfg,
        verbose=0,
    )

    #
    print("Current running case: crcoeff %0.2f, lvcoeff %0.2f, avcoeff %0.4f" %
          (args.crcoeff, args.lvcoeff, args.avcoeff))
    print("Start training...")
    model.learn(total_timesteps=int(5 * 1e7), log_interval=(10, 50), tb_log_name=tb_log_name)


if __name__ == "__main__":
    start = time.time()
    main()
    passed_time = ((time.time() - start) / 60) / 60
    print("Process running time: %f hours" % passed_time)
