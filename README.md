# P3
Code for "[Pareto Policy Pool for Model-based Offline RL](https://openreview.net/forum?id=OqcZu8JIIzS)", presented in ICLR 2022.

## Key Dependencies
```console
python==3.6.13
- d4rl==1.1
- ray==1.0.0
- gym==0.18.3
- torch==1.7.1
- tensorflow==2.3.1
- mujoco-py==2.0.2.13
```

## Quick Start
```python
python p3.py
```
Environment models and pretrained weights can be found in the folder "src/models" and "src/models_params", respectively.

## Acknowledgement
We appreciate the open source of the following projects:

[MOPO](https://github.com/tianheyu927/mopo), [MOReL](https://github.com/aravindr93/mjrl/tree/v2/projects/morel), and [D4RL](https://github.com/Farama-Foundation/D4RL)
