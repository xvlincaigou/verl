---
date: 2026-04-17
question_id: Q002
topics: ["reward", "config", "migration", "backward-compatibility"]
related_files:
  - verl/experimental/reward_loop/__init__.py
  - verl/experimental/reward_loop/reward_loop.py
  - verl/trainer/config/legacy_reward_impl.yaml
  - verl/trainer/config/reward/reward.yaml
  - verl/trainer/main_ppo.py
---

# Question

`migrate_legacy_reward_impl` 这个函数干什么用的？

# Answer

`migrate_legacy_reward_impl` 是用于**向后兼容**的配置迁移函数，将旧的奖励模型配置结构迁移到新的配置结构。

## 背景：配置结构的变化

verl 的奖励模型配置经历了重构，从旧的扁平结构变成了新的嵌套结构：

### 旧配置结构 (legacy_reward_impl.yaml)
```yaml
custom_reward_function:
  path: null
  name: null

reward_model:
  num_workers: null
  reward_manager: null
  enable: null
  model:
    path: null
  rollout:
    name: null
    ...

sandbox_fusion:
  url: null
```

### 新配置结构 (reward.yaml)
```yaml
reward:
  num_workers: 8
  custom_reward_function:
    path: null
    name: compute_score
  reward_manager:
    name: naive
    ...
  reward_model:
    enable: False
    ...
  sandbox_fusion:
    url: null
```

## 迁移逻辑详解

**文件**: `verl/experimental/reward_loop/reward_loop.py:42-97`

```python
def migrate_legacy_reward_impl(config):
    """
    Migrate the legacy reward model implementation to the new one.
    """
    # 1. reward workers 迁移
    # config.reward_model.num_workers -> config.reward.num_workers
    if config.reward_model.num_workers is not None:
        config.reward.num_workers = config.reward_model.num_workers

    # 2. reward manager 迁移
    if config.reward_model.reward_manager is not None:
        config.reward.reward_manager.name = config.reward_model.reward_manager
    
    # 3. custom reward function 迁移
    # config.custom_reward_function -> config.reward.custom_reward_function
    if not all(v is None for v in config.custom_reward_function.values()):
        config.reward.custom_reward_function = config.custom_reward_function

    # 4. reward model 迁移
    for key in ["enable", "enable_resource_pool", "n_gpus_per_node", "nnodes"]:
        if config.reward_model.get(key) is not None:
            config.reward.reward_model[key] = config.reward_model[key]
    
    # 5. sandbox_fusion 迁移
    if not all(v is None for v in config.sandbox_fusion.values()):
        config.reward.sandbox_fusion = config.sandbox_fusion

    # 6. 删除旧配置
    with open_dict(config):
        del config.reward_model
        del config.custom_reward_function
        del config.sandbox_fusion

    return config
```

## 迁移映射表

| 旧配置路径 | 新配置路径 | 说明 |
|-----------|-----------|------|
| `reward_model.num_workers` | `reward.num_workers` | 奖励计算 worker 数量 |
| `reward_model.reward_manager` | `reward.reward_manager.name` | 奖励管理器名称 |
| `custom_reward_function.*` | `reward.custom_reward_function.*` | 自定义奖励函数 |
| `reward_model.enable` | `reward.reward_model.enable` | 是否启用奖励模型 |
| `reward_model.model.path` | `reward.reward_model.model_path` | 模型路径 |
| `reward_model.rollout.*` | `reward.reward_model.rollout.*` | rollout 配置 |
| `sandbox_fusion.*` | `reward.sandbox_fusion.*` | 沙箱配置 |

## 调用时机

**文件**: `verl/trainer/main_ppo.py:46`

```python
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)  # <-- 在这里调用
    run_ppo(config)
```

在 Hydra 解析配置后，立即进行配置迁移，确保后续代码使用新的配置结构。

## 为什么需要这个函数

1. **向后兼容**: 旧用户可以继续使用旧的配置方式
2. **平滑过渡**: 配置重构不会破坏现有脚本
3. **统一接口**: 内部代码只需要处理新的配置结构
4. **配置清理**: 迁移完成后删除旧配置，避免混淆

## Key Points

- 这个函数只在 `main_ppo.py` 入口调用一次
- 使用 `open_dict` 允许修改原本不可变的 OmegaConf 对象
- 旧配置字段会被删除，避免新旧配置共存导致的混淆
- 迁移后所有代码统一访问 `config.reward.*` 路径

## Code References

**函数定义** (`reward_loop.py:42`):
```python
def migrate_legacy_reward_impl(config):
    """Migrate the legacy reward model implementation to the new one."""
```

**调用位置** (`main_ppo.py:46`):
```python
config = migrate_legacy_reward_impl(config)
```

**配置文件**:
- 旧配置: `verl/trainer/config/legacy_reward_impl.yaml`
- 新配置: `verl/trainer/config/reward/reward.yaml`

## Follow-up Questions

- [ ] RewardLoopManager 如何处理不同类型的奖励计算
- [ ] `custom_reward_function` 的完整加载和执行流程
- [ ] `reward_manager` 的不同实现 (naive, advanced 等)
