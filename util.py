import random

import imageio
import numpy as np
import torch
import wandb
import logging
from alg_parameters import *


def set_global_seeds(i):
    """set seed for fair comparison"""
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True


def write_to_tensorboard(global_summary, step, performance_dict=None, mb_loss=None, imitation_loss=None, evaluate=True,
                         greedy=True):
    """record performance using tensorboard"""
    if imitation_loss is not None:
        global_summary.add_scalar(tag='Loss/Imitation_loss', scalar_value=imitation_loss[0], global_step=step)
        global_summary.add_scalar(tag='Grad/Imitation_grad', scalar_value=imitation_loss[1], global_step=step)

        global_summary.flush()
        return
    if evaluate:
        if greedy:
            global_summary.add_scalar(tag='Perf_greedy_eval/Reward', scalar_value=performance_dict['per_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/In_Reward', scalar_value=performance_dict['per_in_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Ex_Reward', scalar_value=performance_dict['per_ex_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Valid_rate', scalar_value=performance_dict['per_valid_rate'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Episode_length', scalar_value=performance_dict['per_episode_len'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Num_block', scalar_value=performance_dict['per_block'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Num_leave_goal',scalar_value=performance_dict['per_leave_goal'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Final_goals', scalar_value=performance_dict['per_final_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Half_goals', scalar_value=performance_dict['per_half_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Block_accuracy', scalar_value=performance_dict['per_block_acc'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Max_goals', scalar_value=performance_dict['per_max_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_greedy_eval/Num_collide', scalar_value=performance_dict['per_num_collide'], global_step=step)

        else:
            global_summary.add_scalar(tag='Perf_random_eval/Reward', scalar_value=performance_dict['per_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/In_Reward', scalar_value=performance_dict['per_in_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Ex_Reward', scalar_value=performance_dict['per_ex_r'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Valid_rate',scalar_value=performance_dict['per_valid_rate'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Episode_length',scalar_value=performance_dict['per_episode_len'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Num_block', scalar_value=performance_dict['per_block'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Num_leave_goal', scalar_value=performance_dict['per_leave_goal'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Final_goals', scalar_value=performance_dict['per_final_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Half_goals', scalar_value=performance_dict['per_half_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Block_accuracy', scalar_value=performance_dict['per_block_acc'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Max_goals', scalar_value=performance_dict['per_max_goals'], global_step=step)
            global_summary.add_scalar(tag='Perf_random_eval/Num_collide', scalar_value=performance_dict['per_num_collide'], global_step=step)

    else:
        loss_vals = np.nanmean(mb_loss, axis=0)
        global_summary.add_scalar(tag='Perf/Reward', scalar_value=performance_dict['per_r'], global_step=step)
        global_summary.add_scalar(tag='Perf/In_Reward', scalar_value=performance_dict['per_in_r'], global_step=step)
        global_summary.add_scalar(tag='Perf/Ex_Reward', scalar_value=performance_dict['per_ex_r'], global_step=step)
        global_summary.add_scalar(tag='Perf/Valid_rate', scalar_value=performance_dict['per_valid_rate'], global_step=step)
        global_summary.add_scalar(tag='Perf/Episode_length',scalar_value=performance_dict['per_episode_len'], global_step=step)
        global_summary.add_scalar(tag='Perf/Num_block', scalar_value=performance_dict['per_block'], global_step=step)
        global_summary.add_scalar(tag='Perf/Num_leave_goal', scalar_value=performance_dict['per_leave_goal'], global_step=step)
        global_summary.add_scalar(tag='Perf/Final_goals', scalar_value=performance_dict['per_final_goals'], global_step=step)
        global_summary.add_scalar(tag='Perf/Half_goals', scalar_value=performance_dict['per_half_goals'], global_step=step)
        global_summary.add_scalar(tag='Perf/Block_accuracy', scalar_value=performance_dict['per_block_acc'], global_step=step)
        global_summary.add_scalar(tag='Perf/Max_goals', scalar_value=performance_dict['per_max_goals'], global_step=step)
        global_summary.add_scalar(tag='Perf/Num_collide', scalar_value=performance_dict['per_num_collide'], global_step=step)
        global_summary.add_scalar(tag='Perf/Rewarded_rate', scalar_value=performance_dict['rewarded_rate'], global_step=step)

        for (val, name) in zip(loss_vals, RecordingParameters.LOSS_NAME):
            if name == 'grad_norm':
                global_summary.add_scalar(tag='Grad/' + name, scalar_value=val, global_step=step)
            else:
                global_summary.add_scalar(tag='Loss/' + name, scalar_value=val, global_step=step)

    global_summary.flush()


def write_to_wandb(step, performance_dict=None, mb_loss=None, imitation_loss=None, evaluate=True, greedy=True):
    """record performance using wandb"""
    print(f"step:{step}:write_to_wandb")
    if imitation_loss is not None:
        wandb.log({'Loss/Imitation_loss': imitation_loss[0]}, step=step)
        wandb.log({'Grad/Imitation_grad': imitation_loss[1]}, step=step)
        return
    if evaluate:
        if greedy:
            wandb.log({'Perf_greedy_eval/Reward': performance_dict['per_r']}, step=step)
            wandb.log({'Perf_greedy_eval/In_Reward': performance_dict['per_in_r']}, step=step)
            wandb.log({'Perf_greedy_eval/Ex_Reward': performance_dict['per_ex_r']}, step=step)
            wandb.log({'Perf_greedy_eval/Valid_rate': performance_dict['per_valid_rate']}, step=step)
            wandb.log({'Perf_greedy_eval/Episode_length': performance_dict['per_episode_len']}, step=step)
            wandb.log({'Perf_greedy_eval/Num_block': performance_dict['per_block']}, step=step)
            wandb.log({'Perf_greedy_eval/Num_leave_goal': performance_dict['per_leave_goal']}, step=step)
            wandb.log({'Perf_greedy_eval/Final_goals': performance_dict['per_final_goals']}, step=step)
            wandb.log({'Perf_greedy_eval/Half_goals': performance_dict['per_half_goals']}, step=step)
            wandb.log({'Perf_greedy_eval/Block_accuracy': performance_dict['per_block_acc']}, step=step)
            wandb.log({'Perf_greedy_eval/Max_goals': performance_dict['per_max_goals']}, step=step)
            wandb.log({'Perf_greedy_eval/Num_collide': performance_dict['per_num_collide']}, step=step)
            wandb.log({'Perf_greedy_eval/Num_tar_comm': performance_dict['per_num_tar_comm']}, step=step)
            wandb.log({'Perf_greedy_eval/Num_comm': performance_dict['per_num_comm']}, step=step)

        else:
            wandb.log({'Perf_random_eval/Reward': performance_dict['per_r']}, step=step)
            wandb.log({'Perf_random_eval/In_Reward': performance_dict['per_in_r']}, step=step)
            wandb.log({'Perf_random_eval/Ex_Reward': performance_dict['per_ex_r']}, step=step)
            wandb.log({'Perf_random_eval/Valid_rate': performance_dict['per_valid_rate']}, step=step)
            wandb.log({'Perf_random_eval/Episode_length': performance_dict['per_episode_len']}, step=step)
            wandb.log({'Perf_random_eval/Num_block': performance_dict['per_block']}, step=step)
            wandb.log({'Perf_random_eval/Num_leave_goal': performance_dict['per_leave_goal']}, step=step)
            wandb.log({'Perf_random_eval/Final_goals': performance_dict['per_final_goals']}, step=step)
            wandb.log({'Perf_random_eval/Half_goals': performance_dict['per_half_goals']}, step=step)
            wandb.log({'Perf_random_eval/Block_accuracy': performance_dict['per_block_acc']}, step=step)
            wandb.log({'Perf_random_eval/Max_goals': performance_dict['per_max_goals']}, step=step)
            wandb.log({'Perf_random_eval/Num_collide': performance_dict['per_num_collide']}, step=step)
            wandb.log({'Perf_random_eval/Num_tar_comm': performance_dict['per_num_tar_comm']}, step=step)
            wandb.log({'Perf_random_eval/Num_comm': performance_dict['per_num_comm']}, step=step)

    else:
        loss_vals = np.nanmean(mb_loss, axis=0)
        wandb.log({'Perf/Reward': performance_dict['per_r']}, step=step)
        wandb.log({'Perf/In_Reward': performance_dict['per_in_r']}, step=step)
        wandb.log({'Perf/Ex_Reward': performance_dict['per_ex_r']}, step=step)
        wandb.log({'Perf/Valid_rate': performance_dict['per_valid_rate']}, step=step)
        wandb.log({'Perf/Episode_length': performance_dict['per_episode_len']}, step=step)
        wandb.log({'Perf/Num_block': performance_dict['per_block']}, step=step)
        wandb.log({'Perf/Num_leave_goal': performance_dict['per_leave_goal']}, step=step)
        wandb.log({'Perf/Final_goals': performance_dict['per_final_goals']}, step=step)
        wandb.log({'Perf/Half_goals': performance_dict['per_half_goals']}, step=step)
        wandb.log({'Perf/Block_accuracy': performance_dict['per_block_acc']}, step=step)
        wandb.log({'Perf/Max_goals': performance_dict['per_max_goals']}, step=step)
        wandb.log({'Perf/Num_collide': performance_dict['per_num_collide']},step=step)
        wandb.log({'Perf/Rewarded_rate': performance_dict['rewarded_rate']},step=step)
        wandb.log({'Perf/Num_tar_comm': performance_dict['per_num_tar_comm']}, step=step)
        wandb.log({'Perf/Num_comm': performance_dict['per_num_comm']}, step=step)

        for (val, name) in zip(loss_vals, RecordingParameters.LOSS_NAME):
            if name == 'grad_norm':
                wandb.log({'Grad/' + name: val}, step=step)
            else:
                wandb.log({'Loss/' + name: val}, step=step)


def make_gif(images, file_name):
    """record gif"""
    imageio.mimwrite(file_name, images, subrectangles=True)
    print("wrote gif")


def reset_env(env, num_agent):
    """reset environment"""
    done = env._reset(num_agent)
    prev_action = np.zeros(num_agent)
    valid_actions = []
    obs = np.zeros((1, num_agent, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE), dtype=np.float32)
    comm_agents = np.zeros((1, num_agent, num_agent), dtype=np.int32)
    vector = np.zeros((1, num_agent, NetParameters.VECTOR_LEN), dtype=np.float32)
    train_valid = np.zeros((num_agent, EnvParameters.N_ACTIONS), dtype=np.float32)

    for i in range(num_agent):
        valid_action = env.list_next_valid_actions(i + 1)
        s = env.observe(i + 1)
        obs[:, i, :, :, :] = s[0]
        vector[:, i, : 3] = s[1]
        comm_agents[:, i, :] = s[2]
        vector[:, i, -1] = prev_action[i]
        valid_actions.append(valid_action)
        train_valid[i, valid_action] = 1
    return done, valid_actions, obs, vector, comm_agents, train_valid


def one_step(env, one_episode_perf, actions, pre_block, model, pre_value, input_state, ps, no_reward, message, comm_agents, tar_comm, episodic_buffer, num_agent):
    """run one step"""
    train_valid = np.zeros((num_agent, EnvParameters.N_ACTIONS), dtype=np.float32)
    obs, vector, comm_agents, rewards, done, next_valid_actions, on_goal, blockings, valid_actions, num_blockings, leave_goals, \
        num_on_goal, max_on_goal, num_collide, action_status, modify_actions, num_comm \
        = env.joint_step(actions, one_episode_perf['num_step'], model, pre_value,
                         input_state, ps, no_reward, message, comm_agents, tar_comm, episodic_buffer)

    one_episode_perf['block'] += num_blockings
    one_episode_perf['num_leave_goal'] += leave_goals
    one_episode_perf['num_collide'] += num_collide
    one_episode_perf['num_target_comm'] += num_comm[1]
    one_episode_perf['num_comm'] += num_comm[2]
    vector[:, :, -1] = modify_actions
    for i in range(num_agent):
        train_valid[i, next_valid_actions[i]] = 1
        if (pre_block[i] < 0.5) == blockings[:, i]:
            one_episode_perf['wrong_blocking'] += 1
    one_episode_perf['num_step'] += 1
    return rewards, next_valid_actions, obs, vector, comm_agents, train_valid, done, blockings, num_on_goal, one_episode_perf, \
        max_on_goal, action_status, modify_actions, on_goal


def update_perf(one_episode_perf, performance_dict, num_on_goals, max_on_goals, num_agent):
    """record batch performance"""
    print(f"one_episode_perf:{one_episode_perf}")
    print(f"before update_perf:{performance_dict}")
    performance_dict['per_ex_r'].append(one_episode_perf['ex_reward'])
    performance_dict['per_in_r'].append(one_episode_perf['in_reward'])
    performance_dict['per_r'].append(one_episode_perf['episode_reward'])
    performance_dict['per_valid_rate'].append(
        ((one_episode_perf['num_step'] * num_agent) - one_episode_perf['invalid']) / (
                one_episode_perf['num_step'] * num_agent))
    performance_dict['per_episode_len'].append(one_episode_perf['num_step'])
    performance_dict['per_block'].append(one_episode_perf['block'])
    performance_dict['per_leave_goal'].append(one_episode_perf['num_leave_goal'])
    performance_dict['per_num_collide'].append(one_episode_perf['num_collide'])
    performance_dict['per_final_goals'].append(num_on_goals)
    performance_dict['per_block_acc'].append(
        ((one_episode_perf['num_step'] * num_agent) - one_episode_perf['wrong_blocking']) / (
                one_episode_perf['num_step'] * num_agent))
    performance_dict['per_max_goals'].append(max_on_goals)
    performance_dict['rewarded_rate'].append(
        one_episode_perf['reward_count'] / (one_episode_perf['num_step'] * num_agent))
    performance_dict['per_num_tar_comm'].append(one_episode_perf['num_target_comm'])
    performance_dict['per_num_comm'].append(one_episode_perf['num_comm'])
    print(f"after update_perf_dict:{performance_dict}")
    return performance_dict


def get_logger():
    logger = logging.getLogger()
    logger.headlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger
    