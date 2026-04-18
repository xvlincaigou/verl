---
date: 2026-04-18
question_id: Q005
topics: ["resource-pool", "ray", "distributed", "gpu-allocation"]
related_files:
  - verl/trainer/main_ppo.py
  - verl/single_controller/ray/base.py
---

# Question

`init_resource_pool_mgr` 函数解释一下。

# Answer

`init_resource_pool_mgr` 是 TaskRunner 中用于**初始化 GPU 资源池管理器**的方法，它定义了如何在 Ray 集群中分配 GPU 资源给不同的 Worker。

## 核心作用

```
┌─────────────────────────────────────────────────────────────────┐
│                  init_resource_pool_mgr                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入: config                                                    │
│    ├── trainer.n_gpus_per_node                                   │
│    ├── trainer.nnodes                                           │
│    ├── reward.reward_model.enable_resource_pool                 │
│    └── distillation.teacher_model.enable_resource_pool          │
│                                                                  │
│  输出: ResourcePoolManager                                       │
│    ├── resource_pool_spec: {"pool_name": [gpus_per_node...]}    │
│    └── mapping: {Role -> "pool_name"}                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 完整代码

**文件**: `verl/trainer/main_ppo.py:221-259`

```python
def init_resource_pool_mgr(self, config):
    """Initialize resource pool manager."""

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }

    # ============ Reward Model 资源池 ============
    if config.reward.reward_model.enable_resource_pool:
        reward_pool = [
            config.reward.reward_model.n_gpus_per_node
        ] * config.reward.reward_model.nnodes
        resource_pool_spec["reward_pool"] = reward_pool
    else:
        # 复用 global_pool 的配置
        config.reward.reward_model.nnodes = config.trainer.nnodes
        config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node

    # ============ Teacher Model 资源池 (Distillation) ============
    if is_distillation_enabled(distillation_config):
        if distillation_config.teacher_model.enable_resource_pool:
            teacher_pool = [
                distillation_config.teacher_model.n_gpus_per_node
            ] * distillation_config.teacher_model.nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool
        else:
            config.distillation.teacher_model.nnodes = config.trainer.nnodes
            config.distillation.teacher_model.n_gpus_per_node = config.trainer.n_gpus_per_node

    # ============ 创建 ResourcePoolManager ============
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=self.mapping  # Role -> Pool 的映射
    )
    return resource_pool_manager
```

## 资源池规格 (resource_pool_spec)

```python
# 示例: 2 节点, 每节点 8 GPU
{
    "global_pool": [8, 8],           # 2 节点, 每节点 8 GPU
    "reward_pool": [4, 4],           # 2 节点, 每节点 4 GPU (可选)
    "teacher_pool": [2, 2],          # 2 节点, 每节点 2 GPU (可选)
}
```

## 三种资源池模式

### 模式 1: 单一资源池 (默认)

所有 Worker 共享同一个资源池：

```yaml
trainer:
  nnodes: 2
  n_gpus_per_node: 8

reward:
  reward_model:
    enable_resource_pool: false  # 默认, 使用 global_pool
```

```
┌─────────────────────────────────────────────────────────────┐
│                      global_pool                            │
│                  [8 GPUs, 8 GPUs] (2 nodes)                  │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ActorRollout  │  │   Critic     │  │ RewardModel  │      │
│  │              │  │              │  │ (if enabled) │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 模式 2: 独立 Reward Model 资源池

Reward Model 使用独立的 GPU 资源：

```yaml
reward:
  reward_model:
    enable: true
    enable_resource_pool: true
    nnodes: 2
    n_gpus_per_node: 4
```

```
┌─────────────────────────────┐    ┌─────────────────────────────┐
│        global_pool          │    │        reward_pool          │
│      [8 GPUs, 8 GPUs]       │    │      [4 GPUs, 4 GPUs]       │
├─────────────────────────────┤    ├─────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ │    │  ┌──────────────────────┐   │
│  │ActorRollout│ │ Critic  │ │    │  │     RewardModel      │   │
│  └──────────┘ └──────────┘ │    │  └──────────────────────┘   │
└─────────────────────────────┘    └─────────────────────────────┘
```

### 模式 3: 独立 Teacher Model 资源池 (Distillation)

```yaml
distillation:
  teacher_model:
    enable_resource_pool: true
    nnodes: 2
    n_gpus_per_node: 2
```

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   global_pool   │  │   reward_pool   │  │  teacher_pool   │
│   [8, 8] GPUs   │  │   [4, 4] GPUs   │  │   [2, 2] GPUs   │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ Actor + Critic  │  │  Reward Model   │  │  Teacher Model  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Role -> Pool 映射

在 `add_*_worker` 方法中设置了 Role 到 Pool 的映射：

```python
def add_actor_rollout_worker(self, config):
    self.mapping[Role.ActorRollout] = "global_pool"

def add_critic_worker(self, config):
    self.mapping[Role.Critic] = "global_pool"

def add_reward_model_resource_pool(self, config):
    if config.reward.reward_model.enable_resource_pool:
        self.mapping[Role.RewardModel] = "reward_pool"
    else:
        self.mapping[Role.RewardModel] = "global_pool"
```

## ResourcePoolManager 创建流程

```python
# 1. TaskRunner.init_resource_pool_mgr 创建
resource_pool_manager = ResourcePoolManager(
    resource_pool_spec={
        "global_pool": [8, 8],
        "reward_pool": [4, 4],
    },
    mapping={
        Role.ActorRollout: "global_pool",
        Role.Critic: "global_pool",
        Role.RewardModel: "reward_pool",
    }
)

# 2. RayPPOTrainer.init_workers() 中调用
resource_pool_manager.create_resource_pool()
# 实际创建 Ray PlacementGroup 资源
```

## ResourcePoolManager 类

**文件**: `verl/single_controller/ray/base.py:182-231`

```python
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]  # pool_name -> [gpus_per_node, ...]
    mapping: dict[int, str]                    # role -> pool_name
    resource_pool_dict: dict[str, RayResourcePool]

    def create_resource_pool(self):
        """创建 Ray 资源池"""
        for pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=3,  # FSDP: 3个 WorkerGroup 共置
                name_prefix=pool_name,
            )
            self.resource_pool_dict[pool_name] = resource_pool

    def get_resource_pool(self, role) -> RayResourcePool:
        """根据 Role 获取对应的资源池"""
        return self.resource_pool_dict[self.mapping[role]]
```

## 为什么需要资源池

1. **资源隔离**: Reward/Teacher Model 可以独立运行，不影响主训练
2. **灵活配置**: 不同组件可以使用不同的 GPU 数量
3. **Ray 调度**: Ray 根据资源池规格分配 PlacementGroup

## Key Points

- `resource_pool_spec`: 定义每个资源池的 GPU 分布 (每节点 GPU 数 × 节点数)
- `mapping`: 定义哪个 Role 使用哪个资源池
- `global_pool`: 默认资源池，用于 Actor/Critic/RefPolicy
- `reward_pool`: 可选，用于独立的 Reward Model
- `teacher_pool`: 可选，用于独立的 Teacher Model (蒸馏)

## Code References

**函数定义** (`main_ppo.py:221`):
```python
def init_resource_pool_mgr(self, config):
    """Initialize resource pool manager."""
```

**ResourcePoolManager** (`single_controller/ray/base.py:182`):
```python
class ResourcePoolManager:
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[int, str]
```

**创建资源池** (`ray_trainer.py:705`):
```python
self.resource_pool_manager.create_resource_pool()
```

## Follow-up Questions

- [ ] RayResourcePool 如何创建 PlacementGroup
- [ ] `max_colocate_count` 参数的含义和最佳实践
- [ ] 多资源池下的进程间通信机制
- [ ] 资源池不足时的错误处理和调度策略
