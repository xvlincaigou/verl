# Topic: verl 架构概览

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           verl Architecture                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐         ┌──────────────────┐                    │
│  │   Entry Script   │────────▶│   main_ppo.py    │                    │
│  │ (shell/config)   │         │  (Hydra + Ray)   │                    │
│  └──────────────────┘         └────────┬─────────┘                    │
│                                        │                                │
│                                        ▼                                │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    Ray Cluster                                │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │    │
│  │  │ ActorRollout │  │    Critic    │  │   RefPolicy  │       │    │
│  │  │   Worker     │  │    Worker    │  │    Worker    │       │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │    │
│  │                                                              │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │    │
│  │  │ RewardModel  │  │   Rollout    │  │  Checkpoint  │       │    │
│  │  │    Worker    │  │   (vLLM)     │  │   Manager    │       │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                      PPO Data Flow                            │    │
│  │                                                              │    │
│  │   Data ──▶ Rollout ──▶ Reward ──▶ Adv ──▶ Update            │    │
│  │    │          │          │        │       │                 │    │
│  │    │          ▼          ▼        ▼       ▼                 │    │
│  │    │      generate    compute  compute  actor/critic        │    │
│  │    │      sequences   reward   adv      update              │    │
│  │    │                                                         │    │
│  │    └──────────────────────────────────────────────────────▶  │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 核心模块

### 1. Trainer 层

- **`RayPPOTrainer`** (`verl/trainer/ppo/ray_trainer.py`): 主训练器
  - 协调所有 worker
  - 实现 PPO 训练循环
  - 管理 checkpoint

### 2. Worker 层

- **`AsyncActorRolloutRefWorker`** (`verl/workers/fsdp_workers.py`)
  - Actor: 策略网络
  - Rollout: 使用 vLLM/SGLang 生成序列
  - Ref: 参考策略 (用于 KL 计算)

- **`CriticWorker`** (`verl/workers/fsdp_workers.py`)
  - Value model
  - 用于 GAE 优势估计

### 3. Rollout 层

- **`AgentLoopManager`** (`verl/experimental/agent_loop.py`)
  - 管理多轮对话
  - 协调 tool 调用

- **`RewardLoopManager`** (`verl/experimental/reward_loop.py`)
  - 计算奖励分数
  - 支持自定义 reward function

### 4. 工具层

- **Tool Support** (`verl/tools/`)
  - `SandboxFusionTool`: 代码执行环境
  - 支持自定义 tool 实现

## Related Questions

- [Q001: verl 工作流程详解](../questions/Q001-verl-workflow-retool.md)
