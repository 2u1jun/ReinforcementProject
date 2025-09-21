#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stand-alone ONNX converter for SB3 PPO models.
- Exports with a REAL input tensor (no numpy path), so web inference works.
- Auto-detects observation shape (vector or image) from the saved policy.
- Puts input name = "observation", output name = "action".
- Adds dynamic batch axis.

Usage:
  python convert_onnx.py --model training_logs/<run>/final_model.zip --out model.onnx
  # or just:
  python convert_onnx.py  # auto-picks the latest final_model.zip under training_logs

Optional:
  --opset 14         (default 14)
  --batch 1          (dummy batch size)
  --obs-shape 4      (override, e.g. "4" or "3,84,84")
"""

import argparse
import os
import sys
import glob
import time
from typing import Tuple

import torch
import torch.nn as nn

try:
    from stable_baselines3 import PPO
except Exception as e:
    print("[ERROR] stable-baselines3가 필요합니다. pip install stable-baselines3", file=sys.stderr)
    raise

# gymnasium은 sb3가 저장한 스페이스를 언패킹할 때 필요할 수 있음
try:
    from gymnasium.spaces import Box, Dict
except Exception:
    # 구버전 gym 사용 환경도 허용
    try:
        from gym.spaces import Box, Dict  # type: ignore
    except Exception as e:
        print("[WARN] gym/gymnasium 미설치. 저장된 모델에서 space를 읽어오지 못할 수 있습니다.", file=sys.stderr)
        Box, Dict = None, None  # type: ignore


class OnnxablePolicy(nn.Module):
    """Torch-only forward pass wrapper over SB3 policy to make ONNX exportable."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy.eval()

    def forward(self, observation: torch.Tensor):
        # SB3 내부 torch 경로: numpy 왕복이 없어 trace 가능
        with torch.no_grad():
            action = self.policy._predict(observation, deterministic=True)
            # SB3는 tensor 반환. 연속이면 (N, act_dim), 이산이면 (N, 1)
            return action


def parse_obs_shape_str(s: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def guess_obs_shape_from_policy(policy) -> Tuple[int, ...]:
    # 저장된 모델에는 observation_space가 들어있음
    space = getattr(policy, "observation_space", None)
    if space is None:
        raise RuntimeError("policy.observation_space를 찾을 수 없습니다. --obs-shape로 직접 지정하세요.")

    shape = getattr(space, "shape", None)
    if not shape:
        # Dict/복합공간이면 여기서 커스텀 필요
        if isinstance(space, Dict):
            raise RuntimeError("Dict observation space는 자동 추정 불가. --obs-shape로 명시하세요.")
        raise RuntimeError("observation_space.shape가 비어 있습니다. --obs-shape로 직접 지정하세요.")

    # 이미지(H, W, C) → CHW 로 바꿔서 더미 만들기 (SB3는 내부에서 CHW로 처리)
    if len(shape) == 3 and shape[-1] in (1, 3, 4):
        h, w, c = shape
        return (c, h, w)

    return tuple(shape)


def find_latest_final_model(base_dir: str = "training_logs") -> str:
    """
    training_logs/**/final_model.zip 중 가장 최근 mtime 파일 반환.
    없으면 체크포인트(rl_model_*.zip)도 후보로 사용.
    """
    patterns = [
        os.path.join(base_dir, "**", "final_model.zip"),
        os.path.join(base_dir, "**", "checkpoints", "*.zip"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat, recursive=True))
    if not candidates:
        return ""
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="", help="SB3 .zip model path (final_model.zip or checkpoint)")
    ap.add_argument("--out", type=str, default="model.onnx", help="Output ONNX path")
    ap.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    ap.add_argument("--batch", type=int, default=1, help="Dummy batch size")
    ap.add_argument("--obs-shape", type=str, default="", help='Override observation shape, e.g. "4" or "3,84,84"')
    args = ap.parse_args()

    model_path = args.model.strip()
    if not model_path:
        model_path = find_latest_final_model()
        if not model_path:
            print("[ERROR] 모델을 찾지 못했습니다. --model로 경로를 지정하거나 training_logs 아래에 final_model.zip이 있는지 확인하세요.")
            sys.exit(1)

    if not os.path.exists(model_path):
        print(f"[ERROR] 모델 파일이 존재하지 않습니다: {model_path}")
        sys.exit(1)

    print(f"[INFO] Loading SB3 PPO model: {os.path.abspath(model_path)}")
    model = PPO.load(model_path, device="cpu")
    policy = model.policy

    # 관측 shape 결정
    if args.obs_shape:
        obs_shape = parse_obs_shape_str(args.obs_shape)
        print(f"[INFO] Using user-provided obs shape: {obs_shape}")
    else:
        obs_shape = guess_obs_shape_from_policy(policy)
        print(f"[INFO] Guessed obs shape from policy: {obs_shape}")

    batch = max(1, int(args.batch))
    dummy = torch.randn((batch,) + tuple(obs_shape), dtype=torch.float32)

    onnxable = OnnxablePolicy(policy)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"[INFO] Exporting to ONNX: {os.path.abspath(out_path)}")
    torch.onnx.export(
        onnxable,
        dummy,
        out_path,
        opset_version=int(args.opset),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
    )

    print("[OK] ONNX export complete.")
    # 빠른 검증(선택): 입력/출력 이름/shape 출력
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        print(f"[VERIFY] inputNames={sess.get_inputs()[0].name}, shape={sess.get_inputs()[0].shape}")
        print(f"[VERIFY] outputNames={sess.get_outputs()[0].name}, shape={sess.get_outputs()[0].shape}")
    except Exception as e:
        print("[NOTE] onnxruntime가 없거나 로드 실패. (웹용 onnxruntime-web에서는 브라우저에서 확인하세요.)")

    # 사용처 도움말
    print("\n[HINT] index.html과 같은 폴더에 'model.onnx'로 두면 브라우저 AI 모드가 즉시 사용합니다.")
    print("       이미지 관측 환경이라면 obs-shape가 (C,H,W)로 나오는지 꼭 확인하세요.")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
