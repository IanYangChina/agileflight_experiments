import argparse
import os
import numpy as np
import torch
from flightgym import VisionEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device
from stable_baselines3.sac.policies import MlpPolicy
from agileflight_policy.common.SAC import SAC
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
    parser.add_argument("--train", type=int, default=1, help="Train the policy or evaluate the policy")
    parser.add_argument("--render", type=int, default=0, help="Render with Unity")
    parser.add_argument("--trial", type=int, default=1, help="SAC trial number")
    parser.add_argument("--iter", type=int, default=100, help="SAC iter number")
    return parser


def main():
    args = parser().parse_args()

    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/sac_debugging"
    os.makedirs(log_dir, exist_ok=True)

    # load configurations
    cfg = YAML().load(
        open(
            rsg_root + "/env_config.yaml", "r"
        )
    )

    cfg["simulation"]["num_envs"] = 100
    cfg["simulation"]["num_threads"] = 3
    cfg["rewards"]["pos_x_coeff"] = -0.001
    cfg["rewards"]["vel_coeff"] = 0.0
    cfg["rewards"]["collision_coeff"] = -0.01
    cfg["rewards"]["angular_vel_coeff"] = 0.0
    cfg["rewards"]["survive_rew"] = 0.03
    # create training environment
    train_env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)

    # set random seed
    configure_random_seed(args.seed, env=train_env)

    if args.render:
        cfg["unity"]["render"] = "yes"

    # create evaluation environment
    old_num_envs = cfg["simulation"]["num_envs"]
    cfg["simulation"]["num_envs"] = 1
    eval_env = wrapper.FlightEnvVec(
        VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    )
    cfg["simulation"]["num_envs"] = old_num_envs

    if args.train:
        model = SAC(
            tensorboard_log=log_dir,
            policy="MlpPolicy",
            policy_kwargs=dict(
                activation_fn=torch.nn.ReLU,
                net_arch=dict(pi=[256, 256], qf=[512, 512]),
                log_std_init=-0.5,
            ),
            env=train_env,
            eval_env=eval_env,
            learning_rate=3e-4,
            tau=0.005,
            learning_starts=1000,
            gradient_steps=100,
            gamma=0.99,
            ent_coef="auto",
            batch_size=256,
            buffer_size=int(1e6),
            train_freq=(1, "step"),
            use_sde=False,  # don't use (gSDE), doesn't work
            env_cfg=cfg,
            verbose=1,
        )
        model.learn(total_timesteps=int(5 * 1e6), log_interval=(50, 10000))
    else:
        os.system(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64 &")
        #
        weight = rsg_root + "/saved/SAC_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
        env_rms = rsg_root +"/saved/SAC_{0}/RMS/iter_{1:05d}.npz".format(args.trial, args.iter)

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
    main()
