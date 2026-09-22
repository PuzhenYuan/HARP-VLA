# MIT License

# Copyright (c) 2021 Oier Mees
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Evaluate HARP-VLA on CALVIN long-horizon sequences."""
import copy
import json
import logging
import os
import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
import calvin_agent
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import count_success, get_env_state_for_initial_condition, get_log_dir
import draccus
import hydra
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored
import torch
from tqdm.auto import tqdm
from moviepy.editor import ImageSequenceClip
from harpvla.policy import load_policy
from harpvla.processing import get_vla_action
logger = logging.getLogger(__name__)

@dataclass
class EvalConfig:
    calvin_root: str = ""
    pretrained_checkpoint: str = "ypz21/HARP_VLA_calvin"
    center_crop: bool = True
    num_images_in_input: int = 2
    use_normalized: bool = True
    use_proprio: bool = True
    unnorm_key: str = "calvin"
    ep_len: int = 360
    num_sequences: int = 1000
    debug: bool = False
    output_dir: str = "outputs/calvin"
    seed: int = 7

def print_and_save(results, sequences, output_dir):
    chain_sr = {i + 1: float(sr) for i, sr in enumerate(count_success(results))}
    succeeded, failed = Counter(), Counter()
    for result, (_, tasks) in zip(results, sequences):
        succeeded.update(tasks[:result])
        if result < len(tasks):
            failed.update([tasks[result]])
    total = succeeded + failed
    report = {
        "avg_seq_len": float(np.mean(results)),
        "chain_sr": chain_sr,
        "task_info": {task: {"success": succeeded[task], "total": total[task]} for task in total},
    }
    (Path(output_dir) / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Average successful sequence length: {report['avg_seq_len']:.3f}")
    for length, rate in chain_sr.items():
        print(f"{length}/5: {rate * 100:.1f}%")


def make_env(dataset_path, observation_space, device):
    val_folder = Path(dataset_path) / "validation"
    from experiments.robot.calvin.calvin_env_wrapper import CalvinEnvWrapperRaw
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env

def evaluate_policy(model, env, eval_dir, ep_len, num_sequences, debug=False):
    conf_dir = Path(calvin_agent.__file__).resolve().parent.parent / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_dir = get_log_dir(eval_dir)
    eval_sequences = get_sequences(num_sequences)

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len)
        results.append(result)
        if not debug:
            success_list = count_success(results)
            with (Path(eval_dir) / 'success_rate.txt').open('a') as f:
                line =f"{sequence_i}/{num_sequences}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += 1
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
            )
        else:
            sequence_i += 1
    print_and_save(results, eval_sequences, eval_dir)
    return results

def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(env, model, task_checker, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def rollout(env, model, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    if debug:
        img_dict = {
            'static': [],
            'gripper': [],
        }

    action_queue = deque()
    for step in range(ep_len):

        # get action chunk
        if len(action_queue) == 0:
            action_queue.extend(model.step(obs, lang_annotation, step))

        action = action_queue.popleft()
        if action[-1] < 0:
            action[-1] = -1
        else:
            action[-1] = 1
        obs, _, _, current_info = env.step(action)

        if debug:
            img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
            img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                for key in img_dict.keys():
                    clip = ImageSequenceClip(img_dict[key], fps=30)
                    clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.gif'), fps=30)
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
        for key in img_dict.keys():
            clip = ImageSequenceClip(img_dict[key], fps=30)
            clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-fail.gif'), fps=30)
    return False

class WrappedCalvinEvaluation(CalvinBaseModel):
    def __init__(self, cfg: EvalConfig, wrapped_model):
        super().__init__()
        self.cfg = cfg
        self.model = wrapped_model

    def reset(self):
        return

    def step(self, obs, instruction, step):
        rgb_obs = obs["rgb_obs"]
        observation = {
            "full_image": rgb_obs["rgb_static"],
            "state": np.array([], dtype=np.float32),
        }

        # If requested, add additional wrist/gripper image for multi-image input
        if self.cfg.num_images_in_input > 1:
            if "rgb_gripper" in rgb_obs:
                observation["wrist_image"] = rgb_obs["rgb_gripper"]

        # Use end-effector pose and gripper state.
        if self.cfg.use_proprio:
            robot_obs = np.asarray(obs["robot_obs"], dtype=np.float32)
            if robot_obs.ndim > 1:
                robot_obs = robot_obs[0]

            if robot_obs.shape[0] >= 7:
                proprio = np.concatenate([robot_obs[:6], robot_obs[[-1]]], axis=0).astype(np.float32)
            else:
                proprio = robot_obs.astype(np.float32)
            observation["state"] = proprio

        actions = get_vla_action(
            self.cfg,
            self.model.vla,
            self.model.processor,
            observation,
            instruction,
            action_head=self.model.action_head,
            proprio_projector=self.model.proprio_projector,
        )
        return actions

@draccus.wrap()
def main(cfg: EvalConfig) -> None:
    if not cfg.calvin_root or not (Path(cfg.calvin_root) / "validation").is_dir():
        raise ValueError("Set --calvin_root to the CALVIN benchmark directory containing validation/.")
    if not torch.cuda.is_available():
        raise RuntimeError("CALVIN evaluation requires an NVIDIA GPU with CUDA and EGL.")
    if cfg.num_sequences <= 0 or cfg.ep_len <= 0:
        raise ValueError("num_sequences and ep_len must be positive.")
    if cfg.num_images_in_input != 2 or not cfg.use_proprio or not cfg.use_normalized:
        raise ValueError("This checkpoint requires two images, proprioception and normalization.")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = load_policy(cfg.pretrained_checkpoint)
    save_path = Path(cfg.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    save_path.mkdir(parents=True, exist_ok=False)
    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": [],
        "state_obs": ["robot_obs"], "actions": ["rel_actions"], "language": ["language"],
    }
    env = make_env(cfg.calvin_root, observation_space, torch.device("cuda:0"))
    try:
        evaluate_policy(
            WrappedCalvinEvaluation(cfg, model), env,
            eval_dir=str(save_path), ep_len=cfg.ep_len,
            num_sequences=cfg.num_sequences, debug=cfg.debug,
        )
    finally:
        env.close()

if __name__ == "__main__":
    main()
