"""CPU checks for observation processing and the evaluation protocol."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from harpvla.processing import normalize_proprio, prepare_images_for_vla
from experiments.robot.calvin.evaluate import EvalConfig, WrappedCalvinEvaluation, print_and_save, rollout


class InferenceTests(unittest.TestCase):
    def test_normalization_preserves_out_of_bounds_values(self):
        stats = {"q01": [-1] * 7, "q99": [1] * 7}
        np.testing.assert_allclose(normalize_proprio(np.array([2.] * 7), stats), [2.] * 7)

    def test_two_camera_preprocessing(self):
        images = [np.zeros((200, 200, 3), np.uint8), np.zeros((84, 84, 3), np.uint8)]
        out = prepare_images_for_vla(images, EvalConfig())
        self.assertEqual([im.size for im in out], [(224, 224), (224, 224)])

    def test_wrapper_extracts_pose_and_last_gripper_value(self):
        model = SimpleNamespace(vla=None, processor=None, action_head=None, proprio_projector=None)
        observation = {"rgb_obs": {"rgb_static": 0, "rgb_gripper": 1}, "robot_obs": np.arange(15)}
        with patch("experiments.robot.calvin.evaluate.get_vla_action", return_value=np.zeros((10, 7))) as call:
            result = WrappedCalvinEvaluation(EvalConfig(), model).step(observation, "open drawer", 0)
            np.testing.assert_array_equal(call.call_args.args[3]["state"], [0, 1, 2, 3, 4, 5, 14])
            self.assertEqual(result.shape, (10, 7))

    def test_result_summary(self):
        sequences = [(None, ['a', 'b', 'c', 'd', 'e'])] * 3
        with tempfile.TemporaryDirectory() as d:
            print_and_save([0, 2, 5], sequences, d)
            result = json.loads((Path(d) / 'result.json').read_text())
            self.assertAlmostEqual(result['avg_seq_len'], 7 / 3)
            self.assertEqual(result['chain_sr']['1'], 2 / 3)
            self.assertEqual(result['task_info']['a'], {'success': 2, 'total': 3})

    def test_rollout_reuses_action_chunk(self):
        class Env:
            def __init__(self): self.steps = 0
            def get_obs(self): return {}
            def get_info(self): return {}
            def step(self, action):
                self.steps += 1
                return {}, 0, False, {"steps": self.steps}
        class Model:
            calls = 0
            def reset(self): pass
            def step(self, *args):
                self.calls += 1
                return np.ones((10, 7))
        class Oracle:
            def get_task_info_for_set(self, start, current, tasks):
                return tasks if current['steps'] == 12 else set()
        env, model = Env(), Model()
        self.assertTrue(rollout(env, model, Oracle(), 'task', {'task': ['instruction']}, False, '.', 0, 0, 20))
        self.assertEqual(model.calls, 2)
        self.assertEqual(env.steps, 12)

if __name__ == '__main__':
    unittest.main()
