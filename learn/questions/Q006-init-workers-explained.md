---
date: 2026-04-18
question_id: Q006
topics: ["ray", "init", "worker", "ppo-trainer"]
related_files:
  - verl/trainer/ppo/ray_trainer.py
---

# Question

讲解一下 `RayPPOTrainer` 的 `init_workers` 这个函数。

# Answer

`init_workers` 是 `RayPPOTrainer` 中**初始化分布式训练 Worker** 的核心函数，负责创建 Ray 资源池、实例化 Worker、初始化各类管理器。

## 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         init_workers()                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Step 1: 创建资源池                                                           │
│  └── resource_pool_manager.create_resource_pool()                           │
│                                                                              │
│  Step 2: 配置 Worker 类                                                       │
│  ├── ActorRollout (RayClassWithInitArgs)                                    │
│  ├── Critic (RayClassWithInitArgs)                                          │
│  └── RefPolicy (RayClassWithInitArgs, optional)                             │
│                                                                              │
│  Step 3: 创建 WorkerGroup (Co-located)                                      │
│  └── create_colocated_worker_cls() → spawn()                                │
│                                                                              │
│  Step 4: 初始化各 Worker                                                      │
│  ├── critic_wg.init_model() / set_loss_fn()                                 │
│  ├── ref_policy_wg.init_model()                                             │
│  └── actor_rollout_wg.init_model()                                          │
│                                                                              │
│  Step 5: 创建管理器                                                           │
│  ├── RewardLoopManager                                                      │
│  ├── TeacherModelManager (optional)                                         │
│  ├── AgentLoopManager                                                       │
│  └── CheckpointEngineManager                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 详细代码解析

**文件**: `verl/trainer/ppo/ray_trainer.py:699-896`

### Step 1: 创建 Ray 资源池

```python
def init_workers(self):
    """Initialize distributed training workers using Ray backend."""
    # 根据 resource_pool_spec 创建 PlacementGroup
    self.resource_pool_manager.create_resource_pool()

    # 初始化 resource_pool -> {role_name: RayClassWithInitArgs} 映射
    self.resource_pool_to_cls = {
        pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
    }
```

### Step 2: 配置 Actor Rollout Worker

```python
# 确定使用 ActorRolloutRef (合并Ref) 还是 ActorRollout
actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout

if self.hybrid_engine:
    actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)

    # RayClassWithInitArgs 包装 Worker 类和初始化参数
    actor_rollout_cls = RayClassWithInitArgs(
        cls=self.role_worker_mapping[actor_role],  # 从 TaskRunner 传入的 Worker 类
        config=self.config.actor_rollout_ref,
        distillation_config=self.config.get("distillation"),
        role=str(actor_role),
    )

    # 将 Worker 类注册到对应资源池
    self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
```

### Step 3: 配置 Critic Worker

```python
if self.use_critic:
    resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

    critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

    # 新实现: 转换为 TrainingWorkerConfig
    if self.use_legacy_worker_impl == "disable":
        critic_cfg = TrainingWorkerConfig(
            model_type="value_model",
            model_config=orig_critic_cfg.model,
            engine_config=engine_config,
            optimizer_config=orig_critic_cfg.optim,
            checkpoint_config=orig_critic_cfg.checkpoint,
        )

    critic_cls = RayClassWithInitArgs(
        cls=self.role_worker_mapping[Role.Critic],
        config=critic_cfg
    )
    self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls
```

### Step 4: 配置 Ref Policy Worker (可选)

```python
if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
    resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
    ref_policy_cls = RayClassWithInitArgs(
        self.role_worker_mapping[Role.RefPolicy],
        config=self.config.actor_rollout_ref,
        role=str(Role.RefPolicy),
    )
    self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls
```

### Step 5: 创建并启动 WorkerGroup

这是最关键的一步，将配置好的 Worker 类实例化为 Ray Actor：

```python
all_wg = {}
wg_kwargs = {"device_name": self.device_name}

# 遍历每个资源池，创建共置的 WorkerGroup
for resource_pool, class_dict in self.resource_pool_to_cls.items():
    if not class_dict:
        continue

    # 将同一资源池内的多个 Worker 类合并为一个类
    # 例如: {"ActorRollout": cls1, "Critic": cls2} 合并为一个类
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)

    # 创建 RayWorkerGroup
    wg_dict = self.ray_worker_group_cls(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
        **wg_kwargs,
    )

    # spawn: 实际创建 Ray Actor 实例
    spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
    all_wg.update(spawn_wg)
```

**`create_colocated_worker_cls` 的作用**:

```
输入: {"ActorRollout": ActorCls, "Critic": CriticCls}
        │
        ▼
合并成一个类，包含两个角色的方法:
        │
        ▼
输出: CoLocatedWorker {
    def ActorRollout__init__(...)
    def ActorRollout_update_actor(...)
    def Critic__init__(...)
    def Critic_update_critic(...)
}
```

### Step 6: 初始化各 Worker

```python
# Critic: 新实现设置 loss function，旧实现调用 init_model
if self.use_critic:
    self.critic_wg = all_wg[str(Role.Critic)]
    if self.use_legacy_worker_impl == "disable":
        self.critic_wg.reset()
        self.critic_wg.set_loss_fn(partial(value_loss, config=orig_critic_cfg))
    else:
        self.critic_wg.init_model()

# Ref Policy: 如果不在 Actor 内，需要单独初始化
if self.use_reference_policy and not self.ref_in_actor:
    if str(Role.RefPolicy) in all_wg:
        self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
        self.ref_policy_wg.init_model()
    else:
        # 新引擎: Ref 在 ActorRolloutRefWorker 内
        self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

# ActorRollout: 最后初始化，让 vLLM 更好估计 KV cache 内存
self.actor_rollout_wg = all_wg[str(actor_role)]
self.actor_rollout_wg.init_model()

if self.ref_in_actor:
    self.ref_policy_wg = self.actor_rollout_wg  # Ref 和 Actor 共享
```

### Step 7: 创建管理器

```python
# Reward Loop Manager: 处理奖励计算
resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
self.reward_loop_manager = RewardLoopManager(
    config=self.config,
    rm_resource_pool=resource_pool,
)

# Teacher Model Manager: 蒸馏用 (可选)
if self.use_teacher_policy:
    teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
    self.teacher_model_manager = TeacherModelManager(
        config=self.config.distillation,
        resource_pool=teacher_resource_pool,
    )

# Agent Loop Manager: 异步 rollout 核心
enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool
reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

self.async_rollout_manager = AgentLoopManager.create(
    config=self.config,
    worker_group=self.actor_rollout_wg,
    rollout_resource_pool=actor_rollout_resource_pool,
    reward_loop_worker_handles=reward_loop_worker_handles,
    teacher_model_manager=self.teacher_model_manager,
)

# Checkpoint Manager: 管理 checkpoint 保存/加载
self.checkpoint_manager = CheckpointEngineManager(
    config=checkpoint_engine_config,
    trainer=self.actor_rollout_wg,
    replicas=self.async_rollout_manager.rollout_replicas,
)

# 让非活跃 replica 进入 sleep 状态，节省内存
self.checkpoint_manager.sleep_replicas()
```

## 关键概念

### 1. `RayClassWithInitArgs`

封装 Worker 类和其初始化参数，延迟到 spawn 时才实例化：

```python
class RayClassWithInitArgs:
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls          # Worker 类 (如 ActorRolloutRefWorker)
        self.args = args        # 位置参数
        self.kwargs = kwargs    # 关键字参数 (config, role, ...)
```

### 2. `create_colocated_worker_cls`

将同一资源池内的多个 Worker 类合并，让它们共置在同一进程中：

```python
# 输入
class_dict = {
    "ActorRollout": ActorRolloutCls,
    "Critic": CriticCls,
}

# 输出: 一个包含所有方法的合并类
CoLocatedWorker = create_colocated_worker_cls(class_dict)
# 这个类可以同时做 Actor 和 Critic 的工作
```

### 3. Worker 初始化顺序

```
1. Critic (if use_critic)
2. RefPolicy (if separate from Actor)
3. ActorRollout (最后，让 vLLM 更好估计内存)
```

### 4. Manager 职责

| Manager | 职责 |
|---------|------|
| `RewardLoopManager` | 管理奖励计算 worker |
| `TeacherModelManager` | 管理蒸馏的教师模型 (可选) |
| `AgentLoopManager` | 管理异步 rollout 流程 |
| `CheckpointEngineManager` | 管理 checkpoint 的保存和加载 |

## Key Points

1. **Co-located Workers**: 同一资源池内的 Worker 会合并为一个类，共享进程
2. **延迟初始化**: `RayClassWithInitArgs` 封装参数，spawn 时才真正创建 Actor
3. **初始化顺序**: Actor 最后初始化，以便 vLLM 准确估计 KV cache 内存
4. **管理器模式**: 各类 Manager 封装特定功能，与 Worker 解耦

## Code References

**函数定义** (`ray_trainer.py:699`):
```python
def init_workers(self):
    """Initialize distributed training workers using Ray backend."""
```

**WorkerGroup 创建** (`ray_trainer.py:780-795`):
```python
worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
wg_dict = self.ray_worker_group_cls(...)
spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
```

**Manager 创建** (`ray_trainer.py:840-896`):
```python
self.reward_loop_manager = RewardLoopManager(...)
self.async_rollout_manager = AgentLoopManager.create(...)
self.checkpoint_manager = CheckpointEngineManager(...)
```

## Follow-up Questions

- [ ] `create_colocated_worker_cls` 的具体实现原理
- [ ] `spawn()` 方法如何创建 Ray Actor
- [ ] `AgentLoopManager.create()` 的详细流程
- [ ] `CheckpointEngineManager` 的 checkpoint 机制
