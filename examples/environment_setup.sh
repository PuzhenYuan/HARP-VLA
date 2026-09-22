#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install --upgrade 'pip<25' 'setuptools<70' wheel
python -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
# urdfpy pins an obsolete networkx; use the compatible version above.
python -m pip install --no-deps urdfpy==0.0.22
mkdir -p third_party
if [[ ! -d third_party/calvin/.git ]]; then
  git clone --recurse-submodules https://github.com/mees/calvin.git third_party/calvin
fi
git -C third_party/calvin checkout fa03f01f19c65920e18cf37398a9ce859274af76
git -C third_party/calvin submodule update --init --recursive
# Install simulator packages without letting legacy requirements replace PyTorch.
python -m pip install --no-deps -e third_party/calvin/calvin_env/tacto
python -m pip install --no-deps -e third_party/calvin/calvin_env
python -m pip install --no-deps -e third_party/calvin/calvin_models
python -m pip install 'git+https://github.com/RLinf/pyfasthash.git@577fab8167020f31349c00d60afea96e7923a023' --no-build-isolation
python -c 'import pyhash; from calvin_agent.evaluation.multistep_sequences import get_sequences; from calvin_env.envs.play_table_env import get_env; from harpvla.policy import load_policy; print("HARP-VLA imports OK")'
