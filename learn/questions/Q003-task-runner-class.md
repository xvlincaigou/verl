---
date: 2026-04-17
question_id: Q003
topics: ["task-runner", "ray", "distributed", "worker-management"]
related_files:
  - verl/trainer/main_ppo.py
  - verl/trainer/ppo/utils.py
  - verl/trainer/ppo/ray_trainer.py
---

# Question

讲解一下 `TaskRunner` 这个类。

# Answer

`TaskRunner` 是 verl 中用于**编排分布式训练任务**的核心类，它负责根据配置创建不同类型的 Worker，并将它们映射到 Ray 资源池。

## 核心作用

```
┌─────────────────────────────────────────────────────────────┐
│                      TaskRunner                              │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │ role_worker_map  │───▶│  ActorRollout    │              │
│  │ (Role → Class)   │    │  Critic          │              │
│  └──────────────────┘    │  RefPolicy       │              │
│                          │  ...             │              │
│  ┌──────────────────┐    └──────────────────┘              │
│  │    mapping       │───▶┌──────────────────┐              │
│  │ (Role → Pool)    │    │  Resource Pool   │              │
│  └──────────────────┘    │  "global_pool"   │              │
│                          │  "reward_pool"   │              │
│                          └──────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## 类定义

**文件**: `verl/trainer/main_ppo.py:108-393`

```python
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}  # Role → Worker Class
        self.mapping = {}              # Role → Resource Pool ID
```

## 核心方法

### 1. `add_actor_rollout_worker(config)`

根据配置添加 Actor + Rollout Worker：

```python
def add_actor_rollout_worker(self, config):
    use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
    
    # 新模型引擎 (use_legacy_worker_impl == "disable")
    if use_legacy_worker_impl == "disable":
        from verl.workers.engine_workers import ActorRolloutRefWorker
        actor_rollout_cls = ActorRolloutRefWorker
        # ...
    
    # 旧模型引擎 - FSDP
    elif config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
        from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
        actor_rollout_cls = AsyncActorRolloutRefWorker
    
    # 旧模型引擎 - Megatron
    elif config.actor_rollout_ref.actor.strategy == "megatron":
        from verl.workers.megatron_workers import AsyncActorRolloutRefWorker
        actor_rollout_cls = AsyncActorRolloutRefWorker
    
    self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
    self.mapping[Role.ActorRollout] = "global_pool"
```

**逻辑**：
- 支持新旧两种 worker 实现 (`use_legacy_worker_impl`)
- 支持多种并行策略 (FSDP/FSDP2/Megatron)
- 如果使用 LoRA，Actor 和 Ref 可以合并为一个 Worker

### 2. `add_critic_worker(config)`

添加 Critic (Value Model) Worker：

```python
def add_critic_worker(self, config):
    if config.critic.strategy in {"fsdp", "fsdp2"}:
        if use_legacy_worker_impl in ["auto", "enable"]:
            from verl.workers.fsdp_workers import CriticWorker
        else:
            from verl.workers.engine_workers import TrainingWorker
            CriticWorker = TrainingWorker  # 新引擎使用通用 TrainingWorker
    
    self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
    self.mapping[Role.Critic] = "global_pool"
```

### 3. `add_ref_policy_worker(config, ref_policy_cls)`

添加 Reference Policy Worker (用于 KL 计算)：

```python
def add_ref_policy_worker(self, config, ref_policy_cls):
    # 新模型引擎中，ref policy 已合并到 ActorRolloutRefWorker
    if use_legacy_worker_impl == "disable":
        return  # 不需要单独创建
    
    if need_reference_policy(config):  # 需要 KL loss 或 KL reward
        self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
        self.mapping[Role.RefPolicy] = "global_pool"
```

### 4. `init_resource_pool_mgr(config)`

初始化资源池管理器：

```python
def init_resource_pool_mgr(self, config):
    resource_pool_spec = {
        "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    
    # 独立的 reward model 资源池
    if config.reward.reward_model.enable_resource_pool:
        reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
        resource_pool_spec["reward_pool"] = reward_pool
    
    # 独立的 teacher model 资源池 (distillation)
    if is_distillation_enabled(distillation_config):
        if distillation_config.teacher_model.enable_resource_pool:
            teacher_pool = [...]
            resource_pool_spec["teacher_pool"] = teacher_pool
    
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, 
        mapping=self.mapping
    )
    return resource_pool_manager
```

### 5. `run(config)` - 主执行流程

```python
def run(self, config):
    # 1. 添加各种 worker
    actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
    self.add_critic_worker(config)
    self.add_reward_model_resource_pool(config)
    self.add_teacher_model_resource_pool(config)
    self.add_ref_policy_worker(config, actor_rollout_cls)
    
    # 2. 验证配置
    validate_config(config, ...)
    
    # 3. 加载模型和 tokenizer
    local_path = copy_to_local(config.actor_rollout_ref.model.path, ...)
    tokenizer = hf_tokenizer(local_path, ...)
    processor = hf_processor(local_path, ...)
    
    # 4. 初始化资源池
    resource_pool_manager = self.init_resource_pool_mgr(config)
    
    # 5. 创建数据集
    train_dataset = create_rl_dataset(...)
    val_dataset = create_rl_dataset(...)
    
    # 6. 初始化 RayPPOTrainer
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=self.role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        ...
    )
    
    # 7. 初始化 worker 并开始训练
    trainer.init_workers()
    trainer.fit()
```

## Role 类型

**文件**: `verl/trainer/ppo/utils.py:27-41`

```python
class Role(Enum):
    Actor = 0          # 策略网络 (旧)
    Rollout = 1        # 推理生成 (旧)
    ActorRollout = 2   # Actor + Rollout 合并
    Critic = 3         # Value 网络
    RefPolicy = 4      # 参考策略 (用于 KL)
    RewardModel = 5    # 奖励模型
    ActorRolloutRef = 6  # Actor + Rollout + Ref 合并
    Env = 7            # 环境 (多模态等)
    TeacherModel = 8   # 教师模型 (蒸馏)
```

## 工作流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                      TaskRunner.run()                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Worker 注册                                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ add_actor_      │ │ add_critic_     │ │ add_ref_policy_ │   │
│  │ rollout_worker  │ │ worker          │ │ worker          │   │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘   │
│           │                   │                   │            │
│           ▼                   ▼                   ▼            │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ role_worker_mapping                                   │     │
│  │   Role.ActorRollout → AsyncActorRolloutRefWorker      │     │
│  │   Role.Critic → CriticWorker                          │     │
│  │   Role.RefPolicy → AsyncActorRolloutRefWorker         │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                  │
│  Step 2: 资源池配置                                                │
│  ┌─────────────────┐ ┌─────────────────┐                        │
│  │ init_resource_  │ │ add_reward_model│                        │
│  │ pool_mgr        │ │ _resource_pool  │                        │
│  └────────┬────────┘ └────────┬────────┘                        │
│           │                   │                                 │
│           ▼                   ▼                                 │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ mapping                                               │     │
│  │   Role.ActorRollout → "global_pool"                   │     │
│  │   Role.Critic → "global_pool"                         │     │
│  │   Role.RewardModel → "reward_pool" (可选)              │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                  │
│  Step 3: 创建 Trainer 并训练                                      │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │ RayPPOTrainer   │───▶│ trainer.fit()   │                    │
│  │ init_workers()  │    │                 │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Points

1. **Worker-Role 映射**: `role_worker_mapping` 存储 Role 到 Worker Class 的映射
2. **资源池映射**: `mapping` 存储 Role 到资源池名称的映射
3. **新旧引擎兼容**: 通过 `use_legacy_worker_impl` 支持新旧两种实现
4. **可选组件**: Reward Model、Ref Policy、Teacher Model 都是可选的
5. **Ray Remote**: 所有 Worker 都是 `ray.remote()` 装饰后的类

## Code References

**TaskRunner 定义** (`main_ppo.py:108`):
```python
class TaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}
```

**Role 枚举** (`ppo/utils.py:27`):
```python
class Role(Enum):
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    ...
```

**Worker 创建** (`main_ppo.py:300`):
```python
actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
self.add_critic_worker(config)
self.add_reward_model_resource_pool(config)
```

## Follow-up Questions

- [ ] ResourcePoolManager 如何分配 GPU 资源
- [ ] `ray.remote()` 后的 Worker 类如何实际创建实例
- [ ] 新旧 worker 实现 (`fsdp_workers.py` vs `engine_workers.py`) 的区别
- [ ] `trainer.init_workers()` 的详细初始化流程
