# PR17 截止时间 pilot

原四臂 100 步流程按实测无法在 2026-10-14 10:00（Asia/Shanghai）前完成，因此切换为时间盒 pilot：

- 四卡布局、Qwen3-4B、关闭 thinking、8192 token 上限、每步 32 条轨迹均保持不变。
- 统一使用 20 步；LoRA 两臂复用已经完成的 step20 检查点，core8 两臂新跑 20 步。
- 每臂评测 64 道 MATH-500、每题 4 次；删除重复的四次初始检查点评测。
- 该 pilot 用于在截止时间前检查方向和接线，不能当作原计划 100 步正式结果。

当前后台状态见 [`deadline-pilot-status.json`](../../operations/pr17-20261013/deadline-pilot-status.json)。
