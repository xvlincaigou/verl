---
date: 2026-04-18
question_id: Q004
topics: ["worker", "engine", "fsdp", "implementation"]
related_files:
  - verl/trainer/main_ppo.py
  - verl/workers/engine_workers.py
  - verl/workers/fsdp_workers.py
---

# Question

`add_actor_rollout_worker` 支持新旧两种 worker 实现？哪个是新哪个是旧？

# Answer

## 两种实现概览

| 实现 | 文件 | 类名 | 控制方式 |
|-----|------|------|---------|
| **新实现** | `verl/workers/engine_workers.py` | `ActorRolloutRefWorker` | `use_legacy_worker_impl="disable"` |
| **旧实现** | `verl/workers/fsdp_workers.py` | `AsyncActorRolloutRefWorker` | `use_legacy_worker_impl="auto"` 或 `"enable"` |

## 代码中的判断逻辑

**文件**: `verl/trainer/main_ppo.py:124-176`

```python
def add_actor_rollout_worker(self, config):
    use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
    
    # ==================== 新实现 ====================
    if use_legacy_worker_impl == "disable":
        from verl.workers.engine_workers import ActorRolloutRefWorker
        actor_rollout_cls = ActorRolloutRefWorker
        # ...
        return actor_rollout_cls, ray_worker_group_cls
    
    # ==================== 旧实现 ====================
    if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
        from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
        actor_rollout_cls = AsyncActorRolloutRefWorker
        
    elif config.actor_rollout_ref.actor.strategy == "megatron":
        from verl.workers.megatron_workers import AsyncActorRolloutRefWorker
        actor_rollout_cls = AsyncActorRolloutRefWorker
```

## 核心区别

### 1. 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     新实现 (engine_workers)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │  TrainingWorker │ ◄── 通用基础类                            │
│  │  (基础训练功能)  │                                            │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │ActorRolloutRefWorker│ ◄── 继承自 TrainingWorker             │
│  │(Actor+Rollout+Ref) │     更通用、模块化                      │
│  └─────────────────┘                                            │
│                                                                  │
│  特点:                                                           │
│  - 使用 EngineRegistry 管理不同引擎                              │
│  - 统一的 TrainingWorkerConfig 配置                              │
│  - 更好的抽象，支持多种后端 (FSDP/Megatron/...)                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     旧实现 (fsdp_workers)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │     Worker      │ ◄── Ray Worker 基类                        │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │ActorRolloutRefWorker│ ◄── 专门的 FSDP 实现                   │
│  │  (旧版实现)      │     包含大量 FSDP 特定逻辑                 │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │AsyncActorRolloutRefWorker│ ◄── 异步版本                      │
│  └─────────────────┘                                            │
│                                                                  │
│  特点:                                                           │
│  - 直接在类中处理 FSDP 细节                                      │
│  - 包含很多特定的优化逻辑                                        │
│  - 代码相对复杂，耦合度高                                        │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 配置方式对比

**旧实现配置** (`fsdp_workers.py`):
```python
# 专门的 FSDP 配置类
from verl.workers.config import FSDPActorConfig, FSDPEngineConfig

class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, config: DictConfig):
        # 直接处理 FSDP 初始化
        self.fsdp_config = config.actor.fsdp_config
        self.offload = self.fsdp_config.get("param_offload", False)
        # ... 大量 FSDP 特定逻辑
```

**新实现配置** (`engine_workers.py`):
```python
# 通用的 TrainingWorkerConfig
from verl.workers.config import TrainingWorkerConfig

class ActorRolloutRefWorker(Worker):
    def __init__(self, config: TrainingWorkerConfig):
        from verl.workers.engine import BaseEngine, EngineRegistry
        # 通过 EngineRegistry 获取引擎
        self.engine = EngineRegistry.get_engine_class(engine_type)(...)
```

### 3. 关键差异总结

| 方面 | 旧实现 (fsdp_workers) | 新实现 (engine_workers) |
|-----|----------------------|------------------------|
| **设计理念** | 专门化，针对 FSDP 优化 | 通用化，引擎抽象 |
| **代码复杂度** | 高，FSDP 逻辑耦合 | 低，模块化设计 |
| **扩展性** | 难扩展，需改多个类 | 易扩展，添加新引擎即可 |
| **配置方式** | OmegaConf DictConfig | 强类型 dataclass |
| **支持后端** | 主要是 FSDP | FSDP/Megatron/更多 |
| **维护状态** | 维护中，逐渐迁移 | 主要发展方向 |

## 为什么有两个实现

1. **平滑过渡**: 旧实现稳定运行，新实现逐步完善
2. **向后兼容**: 现有用户脚本无需修改即可继续运行
3. **风险规避**: 新实现经过充分测试后再成为默认

## 如何选择

```yaml
# 使用新实现 (推荐用于新项目)
trainer:
  use_legacy_worker_impl: "disable"

# 使用旧实现 (默认，兼容现有项目)
trainer:
  use_legacy_worker_impl: "auto"  # 或 "enable"
```

## Key Points

- **新实现**: `engine_workers.py` 中的 `ActorRolloutRefWorker`，更通用、模块化
- **旧实现**: `fsdp_workers.py` 中的 `AsyncActorRolloutRefWorker`，专门针对 FSDP
- 通过 `use_legacy_worker_impl` 配置切换
- 新实现使用 `EngineRegistry` 管理不同后端引擎
- 旧实现包含大量 FSDP 特定的优化细节

## Code References

**新旧判断逻辑** (`main_ppo.py:130-131`):
```python
if use_legacy_worker_impl == "disable":
    from verl.workers.engine_workers import ActorRolloutRefWorker  # 新
else:
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker  # 旧
```

**新实现类定义** (`engine_workers.py:436`):
```python
class ActorRolloutRefWorker(Worker, DistProfilerExtension):
```

**旧实现类定义** (`fsdp_workers.py:1764`):
```python
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
```

## Follow-up Questions

- [ ] `EngineRegistry` 如何管理不同引擎
- [ ] 新实现的 `TrainingWorker` 具体架构
- [ ] 两种实现的性能对比
- [ ] 完全迁移到新实现的时间计划
