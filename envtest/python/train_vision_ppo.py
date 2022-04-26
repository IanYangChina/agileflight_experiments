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
    parser.add_argument("--level", type=str, default="medium", help="Task level: easy, medium, hard")
    parser.add_argument("--collision_terminal_coeff", type=float, default=0.01, help="Env collision terminal reward coefficient")
    parser.add_argument("--crcoeff", type=float, default=0.0, help="Env collision reward coefficient")
    parser.add_argument("--lvcoeff", type=float, default=0.0, help="Env linear velocity penalty coefficient")
    parser.add_argument("--posxcoeff", type=float, default=0.01, help="Env linear velocity penalty coefficient")
    parser.add_argument("--avcoeff", type=float, default=0.0, help="Env angular velocity penalty coefficient")
    return parser


def main():
    args = parser().parse_args()

    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + '/' + args.level + '_'\
              'collision_terminal_coeff_' + str(args.collision_terminal_coeff) + \
              '_crcoeff_' + str(args.crcoeff) + \
              '_posxcoeff_' + str(args.posxcoeff)
    tb_log_name = 'PPO'
    os.makedirs(log_dir, exist_ok=True)

    # load configurations
    cfg = YAML().load(
        open(
            rsg_root + "/env_config.yaml", "r"
        )
    )

    # create training environment
    cfg["environment"]["level"] = args.level
    cfg["rewards"]["collision_terminal_coeff"] = float(args.collision_terminal_coeff)
    cfg["rewards"]["pos_x_coeff"] = float(args.posxcoeff)
    cfg["rewards"]["vel_coeff"] = 0.0
    cfg["rewards"]["collision_coeff"] = float(args.crcoeff)
    cfg["rewards"]["angular_vel_coeff"] = 0.0
    cfg["rewards"]["survive_rew"] = 0.0
    cfg["simulation"]["num_threads"] = 6
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
        verbose=1,
    )

    #
    print("Current running case: collision_terminal_coeff %0.2f, crcoeff %0.2f, posxcoeff %0.2f" %
          (args.collision_terminal_coeff, args.crcoeff, args.posxcoeff))
    print("Start training...")
    model.learn(total_timesteps=int(5 * 1e7), log_interval=(10, 50), tb_log_name=tb_log_name)


if __name__ == "__main__":
    start = time.time()
    main()
    passed_time = ((time.time() - start) / 60) / 60
    print("Process running time: %f hours" % passed_time)
