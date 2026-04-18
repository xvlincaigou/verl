# verl 学习笔记

## 学习进度

| 日期 | 内容 | 文件 |
|-----|------|------|
| 2026-04-16 | verl 工作流程详解 (以 retool 为例) | [Q001](questions/Q001-verl-workflow-retool.md) |
| 2026-04-17 | migrate_legacy_reward_impl 函数作用 | [Q002](questions/Q002-migrate-legacy-reward-impl.md) |
| 2026-04-17 | TaskRunner 类详解 | [Q003](questions/Q003-task-runner-class.md) |
| 2026-04-18 | Worker 新旧实现对比 | [Q004](questions/Q004-worker-impl-new-vs-legacy.md) |
| 2026-04-18 | init_resource_pool_mgr 详解 | [Q005](questions/Q005-init-resource-pool-mgr.md) |
| 2026-04-18 | RayPPOTrainer.init_workers 详解 | [Q006](questions/Q006-init-workers-explained.md) |

## 主题索引

### 架构
- [Architecture 概览](topics/architecture.md)

### 奖励系统
- [Reward 配置迁移](questions/Q002-migrate-legacy-reward-impl.md)

## 问题列表

### Q001: verl 工作流程详解
**文件**: `recipe/retool/run_qwen2-32b_dapo.sh`

以 retool 为例，讲解了从 shell 入口到 PPO 训练完成的完整流程：
1. 入口脚本配置
2. Ray 集群初始化
3. Worker 创建与资源分配
4. PPO 训练循环 (8 个步骤)
5. 多轮 Tool Use 特殊流程

## 待深入主题

- [ ] RayPPOTrainer 初始化细节
- [ ] AgentLoopManager 多轮对话实现
- [ ] Rollout (vLLM/SGLang) 内部机制
- [ ] Checkpoint 保存/加载机制
- [ ] Advantage Estimator 实现 (GAE/GRPO/REINFORCE++)
