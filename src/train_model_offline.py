import gym
gym.logger.setLevel(40)
import d4rl
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from models.constructor import construct_model, format_samples_for_training


def model_name(args):
    name = f'{args.env}-{args.quality}'
    if args.separate_mean_var:
        name += '_smv'
    name += f'_{args.seed}'
    return name


def main(args):
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    # tf.random.set_seed(args.seed)

    if args.quality == "mixed":
        env_1 = gym.make(f'{args.env}-random-v0')
        env_2 = gym.make(f'{args.env}-medium-v0')
        env_3 = gym.make(f'{args.env}-expert-v0')
        dataset_1 = d4rl.qlearning_dataset(env_1)
        dataset_2 = d4rl.qlearning_dataset(env_2)
        dataset_3 = d4rl.qlearning_dataset(env_3)
        obs_dim = dataset_1['observations'].shape[1]
        act_dim = dataset_1['actions'].shape[1]
        # print(dataset_1['observations'].shape, dataset_2['observations'].shape)
        obs = np.concatenate((dataset_1['observations'], dataset_2['observations'], dataset_3['observations']), axis=0)
        acts = np.concatenate((dataset_1['actions'], dataset_2['actions'], dataset_3['actions']), axis=0)
        next_obs = np.concatenate((dataset_1['next_observations'], dataset_2['next_observations'], dataset_3['next_observations']), axis=0)
        dataset_1['rewards'] = np.expand_dims(dataset_1['rewards'], 1)
        dataset_2['rewards'] = np.expand_dims(dataset_2['rewards'], 1)
        dataset_3['rewards'] = np.expand_dims(dataset_3['rewards'], 1)
        res = np.concatenate((dataset_1['rewards'], dataset_2['rewards'], dataset_3['rewards']), axis=0)
        # if args.env == 'hopper':
        #     dataset_4 = np.load('/home/yijunyan/Data/PyCode/ModelARS/hopper_expert/hopper_expert_addition.npy', allow_pickle=True).item()
        #     obs = np.concatenate((obs, dataset_4['observations']), axis=0)
        #     acts = np.concatenate((acts, dataset_4['actions']), axis=0)
        #     dataset_4['rewards'] = np.expand_dims(dataset_4['rewards'], 1)
        #     res = np.concatenate((res, dataset_4['rewards']), axis=0)
    elif args.quality == "random":
        env_1 = gym.make(f'{args.env}-random-v0')
        dataset_1 = d4rl.qlearning_dataset(env_1)
        obs_dim = dataset_1['observations'].shape[1]
        act_dim = dataset_1['actions'].shape[1]
        obs = dataset_1['observations']
        acts = dataset_1['actions']
        next_obs = dataset_1['next_observations']
        res = np.expand_dims(dataset_1['rewards'], 1)

    elif args.quality == "medium":
        env_1 = gym.make(f'{args.env}-medium-v0')
        dataset_1 = d4rl.qlearning_dataset(env_1)
        obs_dim = dataset_1['observations'].shape[1]
        act_dim = dataset_1['actions'].shape[1]
        obs = dataset_1['observations']
        acts = dataset_1['actions']
        next_obs = dataset_1['next_observations']
        res = np.expand_dims(dataset_1['rewards'], 1)

    elif args.quality == "med-expert":
        env_2 = gym.make(f'{args.env}-medium-v0')
        env_3 = gym.make(f'{args.env}-expert-v0')
        dataset_2 = d4rl.qlearning_dataset(env_2)
        dataset_3 = d4rl.qlearning_dataset(env_3)
        obs_dim = dataset_2['observations'].shape[1]
        act_dim = dataset_2['actions'].shape[1]
        obs = np.concatenate((dataset_2['observations'], dataset_3['observations']), axis=0)
        acts = np.concatenate((dataset_2['actions'], dataset_3['actions']), axis=0)
        next_obs = np.concatenate((dataset_2['next_observations'], dataset_3['next_observations']), axis=0)
        dataset_2['rewards'] = np.expand_dims(dataset_2['rewards'], 1)
        dataset_3['rewards'] = np.expand_dims(dataset_3['rewards'], 1)
        res = np.concatenate((dataset_2['rewards'], dataset_3['rewards']), axis=0)
    elif args.quality == "med-replay":
        env_1 = gym.make(f'{args.env}-medium-replay-v0')
        dataset_1 = d4rl.qlearning_dataset(env_1)
        obs_dim = dataset_1['observations'].shape[1]
        act_dim = dataset_1['actions'].shape[1]
        obs = dataset_1['observations']
        acts = dataset_1['actions']
        next_obs = dataset_1['next_observations']
        res = np.expand_dims(dataset_1['rewards'], 1)
    else:
        raise NotImplementedError

    model = construct_model(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.hidden_dim,
                            num_networks=args.num_networks, num_elites=args.num_elites,
                            model_type=args.model_type, separate_mean_var=args.separate_mean_var,
                            name=model_name(args), save_dir=args.model_dir, deterministic=False)
    
    train_inputs, train_outputs = format_samples_for_training(obs, acts, next_obs, res)
    model.train(train_inputs, train_outputs,
                batch_size=args.batch_size, holdout_ratio=args.holdout_ratio,
                max_epochs=args.max_epochs, max_t=args.max_t)
    model.save(args.model_dir, 0)


if __name__ == '__main__':
    from argparse import ArgumentParser
    import warnings
    warnings.filterwarnings("ignore")
    parser = ArgumentParser()
    parser.add_argument('--env', required=True)
    parser.add_argument('--quality', required=True)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--model-type', default='mlp')
    parser.add_argument('--separate-mean-var', default=True)
    parser.add_argument('--num-networks', default=7, type=int)
    parser.add_argument('--num-elites', default=5, type=int)
    parser.add_argument('--hidden-dim', default=200, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--holdout-ratio', default=0.2, type=float)
    parser.add_argument('--max-epochs', default=200, type=int)
    parser.add_argument('--max-t', default=None, type=float)
    parser.add_argument('--model-dir', default='/home/yijunyan/Data/Datasets/d4rl/models_v2')
    main(parser.parse_args())