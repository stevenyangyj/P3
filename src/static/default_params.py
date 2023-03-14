import argparse

def get_params():

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v3')
    parser.add_argument('--fake_env_name', type=str, default='halfcheetah')
    parser.add_argument('--quality', type=str, default='random')
    parser.add_argument('--u_coeff', type=float, default=0.0)
    parser.add_argument('--mgda', type=bool, default=True)
    parser.add_argument('--num_pareto', type=int, default=100)
    parser.add_argument('--turn_tfboard', type=bool, default=False)
    parser.add_argument('--n_iter', '-n', type=int, default=200)
    parser.add_argument('--outer_iter', type=int, default=5)
    parser.add_argument('--select_type', type=str, default='rank')
    parser.add_argument('--regularity', type=str, default='kl_con')
    parser.add_argument('--optim', type=str, default='Adam')

    parser.add_argument('--deltas_used', '-du', type=int, default=20) # no use
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--delta_std', '-std', type=float, default=0.03)
    parser.add_argument('--psi', type=float, default=-1e-2)
    parser.add_argument('--l2_coeff', type=float, default=0.0)
    parser.add_argument('--annealing', type=bool, default=False)

    parser.add_argument('--n_directions', '-nd', type=int, default=30)
    parser.add_argument('--n_workers', '-e', type=int, default=30)
    # should be equal to 1000 due to reference vector [expert score, 1000*exp(0)]
    parser.add_argument('--rollout_length', '-r', type=int, default=1000)
    parser.add_argument('--smooth_num', type=int, default=3)

    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--policy_type', type=str, default='mlp')
    parser.add_argument('--bc_init', type=bool, default=True)
    parser.add_argument('--dir_path', type=str, default='d4rl_exp_tries')

    # for ARS V1 use filter = 'NoFilter'
    # 'MeanStdFilter' 'FixedFilter'
    parser.add_argument('--filter', type=str, default='FixedFilter')

    args = parser.parse_args()
    params = vars(args)

    return params