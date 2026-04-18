---
date: 2026-04-16
question_id: Q001
topics: ["workflow", "ppo", "ray", "retool", "multi-turn"]
related_files:
  - recipe/retool/run_qwen2-32b_dapo.sh
  - verl/trainer/main_ppo.py
  - verl/trainer/ppo/ray_trainer.py
  - recipe/retool/retool.py
  - recipe/retool/sandbox_fusion_tool_config.yaml
---

# Question

以 `recipe/retool/run_qwen2-32b_dapo.sh` 为例子，从入口开始讲解整个 verl 的工作流程。

# Answer

以 `recipe/retool/run_qwen2-32b_dapo.sh` 为例，verl 的工作流程可以分为以下几个阶段：

## 1. 入口脚本 (Shell Script)

**文件**: `recipe/retool/run_qwen2-32b_dapo.sh`

这是训练任务的入口，主要完成：

```bash
# 1. 配置数据和模型路径
train_files="['$dapo_math_17k']"
test_files="['$aime_2025']"

# 2. 配置算法参数 (GRPO)
adv_estimator=grpo
use_kl_in_reward=False

# 3. 配置多轮对话参数
max_turns=8

# 4. 启动训练 - 调用 main_ppo 模块
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files="$train_files" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    ...
```

关键点：
- 使用 Hydra 配置管理，通过命令行参数覆盖默认配置
- 启用多轮对话 (`multi_turn.enable=True`)
- 配置 tool 使用 (`tool_config_path`)
- 指定自定义 dataset 和 reward function

---

## 2. 主入口 (Main Entry)

**文件**: `verl/trainer/main_ppo.py`

### 2.1 主函数流程

```python
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)  # 自动检测 NPU/CUDA
    config = migrate_legacy_reward_impl(config)
    run_ppo(config)
```

### 2.2 Ray 集群初始化

```python
def run_ppo(config, task_runner_class=None):
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
```

### 2.3 TaskRunner 角色映射

```python
class TaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}  # Role -> Worker Class
        self.mapping = {}              # Role -> Resource Pool
```

Role 定义：
- `ActorRollout` / `ActorRolloutRef`:  actor 和 rollout worker (可合并)
- `Critic`: Value model worker
- `RefPolicy`: Reference policy worker (用于 KL 计算)
- `RewardModel`: 奖励模型 worker

---

## 3. Worker 初始化流程

### 3.1 添加 Actor Rollout Worker

```python
def add_actor_rollout_worker(self, config):
    # 根据策略选择 Worker 类
    if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
        from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
        actor_rollout_cls = AsyncActorRolloutRefWorker
```

### 3.2 资源池管理

```python
def init_resource_pool_mgr(self, config):
    resource_pool_spec = {
        "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    # 如果有独立的 reward model 资源池
    if config.reward.reward_model.enable_resource_pool:
        resource_pool_spec["reward_pool"] = reward_pool
```

### 3.3 Worker Group 创建

在 `RayPPOTrainer.init_workers()` 中：

```python
for resource_pool, class_dict in self.resource_pool_to_cls.items():
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    wg_dict = self.ray_worker_group_cls(...)
    spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
```

---

## 4. 核心训练流程

**文件**: `verl/trainer/ppo/ray_trainer.py`, `RayPPOTrainer.fit()`

### 4.1 训练循环概览

```python
def fit(self):
    # 1. 加载 checkpoint
    self._load_checkpoint()
    
    # 2. 初始化 validation
    if self.config.trainer.get("val_before_train", True):
        val_metrics = self._validate()
    
    # 3. 训练循环
    for epoch in range(current_epoch, self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            # PPO 数据流
            self._train_step(batch_dict)
```

### 4.2 单步训练流程 (PPO Data Flow)

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Generate Sequences (Rollout)                        │
│  - async_rollout_manager.generate_sequences()               │
│  - 使用 vLLM/SGLang 生成响应                                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Compute Reward                                      │
│  - reward_loop_manager 计算奖励                              │
│  - 对于 retool: 调用自定义 reward function                   │
│    (recipe/retool/retool.py::compute_score)                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Compute Old Log Probs                               │
│  - actor_rollout_wg.compute_log_prob()                      │
│  - 用于 PPO 的重要性采样比率计算                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Compute Reference Log Probs (可选)                  │
│  - ref_policy_wg.compute_ref_log_prob()                     │
│  - 用于 KL 惩罚计算                                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Compute Values (Critic, 可选)                       │
│  - critic_wg.compute_values()                               │
│  - 用于 GAE 优势估计                                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 6: Compute Advantages                                  │
│  - compute_advantage()                                      │
│  - 支持 GAE / GRPO / REINFORCE++ 等                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 7: Update Actor                                        │
│  - actor_rollout_wg.update_actor()                          │
│  - PPO 策略更新                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 8: Update Critic (可选)                                │
│  - critic_wg.update_critic()                                │
│  - Value function 更新                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Retool 特殊流程 (多轮 Tool Use)

### 5.1 Agent Loop Manager

**文件**: `verl/experimental/agent_loop.py`

```python
# 创建 AgentLoopManager
self.async_rollout_manager = AgentLoopManager.create(
    config=self.config,
    worker_group=self.actor_rollout_wg,
    reward_loop_worker_handles=reward_loop_worker_handles,
    ...
)
```

### 5.2 Tool 配置

**文件**: `recipe/retool/sandbox_fusion_tool_config.yaml`

```yaml
tools:
  - class_name: "recipe.retool.retool.CustomSandboxFusionTool"
    config:
      sandbox_fusion_url: "http://localhost:8080/run_code"
      num_workers: 128
    tool_schema:
      type: "function"
      function:
        name: "code_interpreter"
```

### 5.3 多轮对话流程

```
User Prompt
    ↓
┌─────────────────┐
│  LLM Generation │  ← 生成初始响应
└─────────────────┘
    ↓ (如果需要 tool call)
┌─────────────────┐
│  Tool Execution │  ← 调用 sandbox 执行代码
│  (SandboxFusion)│
└─────────────────┘
    ↓
┌─────────────────┐
│  Observation    │  ← 返回执行结果
└─────────────────┘
    ↓
┌─────────────────┐
│  LLM Generation │  ← 基于 observation 继续生成
└─────────────────┘
    ↓ (重复直到 max_turns 或不需要 tool)
Final Answer
```

### 5.4 自定义 Reward Function

**文件**: `recipe/retool/retool.py`

```python
def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    # 1. 使用 math_dapo 计算基础分数
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
    
    # 2. 多轮对话奖励调整
    num_turns = extra_info["num_turns"]
    if result["score"] < 0:
        # 鼓励模型使用 tool
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)
    
    return result
```

### 5.5 自定义 Dataset

```python
class CustomRLHFDataset(RLHFDataset):
    def map_fn(self, row: dict, *, data_source: str = None):
        # 添加 answer format 要求
        prompt = problem + answer_format  # \boxed{...}
        data = {
            "prompt": [{"role": "user", "content": prompt}],
            "agent_name": "tool_agent",  # 标记使用 tool agent
            ...
        }
        return data
```

---

## 6. 关键组件总结

| 组件 | 文件 | 职责 |
|-----|------|------|
| ActorRolloutRefWorker | `verl/workers/fsdp_workers.py` | 模型推理、训练、rollout |
| CriticWorker | `verl/workers/fsdp_workers.py` | Value model 计算 |
| AgentLoopManager | `verl/experimental/agent_loop.py` | 多轮对话管理 |
| RewardLoopManager | `verl/experimental/reward_loop.py` | 奖励计算 |
| DataProto | `verl/protocol.py` | 数据传输协议 |

## Key Points

1. **Ray-based 分布式**: verl 使用 Ray 作为分布式计算框架，所有 worker 都是 Ray actor
2. **Hydra 配置管理**: 通过 Hydra 实现灵活的配置覆盖
3. **Async Rollout**: 使用 vLLM/SGLang 异步生成序列
4. **Multi-turn Support**: 通过 AgentLoopManager 支持多轮 tool use
5. **Customizable**: 支持自定义 dataset、reward function、tool

## Code References

**PPO 训练数据流** (`ray_trainer.py:1350+`):
```python
# 生成序列
gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

# 计算奖励
reward_tensor, reward_extra_infos_dict = extract_reward(batch)

# 计算 old log probs
old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)

# 计算 advantages
batch = compute_advantage(...)

# 更新 actor
actor_output = self._update_actor(batch)

# 更新 critic
critic_output = self._update_critic(batch)
```

## Follow-up Questions

- [ ] RayPPOTrainer 的完整初始化流程
- [ ] AgentLoopManager 如何处理多轮对话的 context 传递
- [ ] vLLM 和 SGLang rollout 的具体区别
- [ ] Checkpoint 保存和加载的详细机制
