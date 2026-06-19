# KV Cache Learned Cache 仿真实验

这个项目用于计算机系统结构课程论文。这个项目用于仿真大语言模型推理中的 KV Cache 管理策略。程序会生成 KV block 访问序列，并比较 FIFO、LRU、LFU、Heuristic 和 Learned Cache 五种策略。

当前版本已经对齐题目中的三个关键词：

- **重用距离预测**：Learned Cache 使用 `DecisionTreeRegressor` 预测某个 KV block 距离下一次访问还有多远。
- **动态驱逐**：cache 满时，优先驱逐预测重用距离较远、规则重要性较低、head activity 较低的 block。
- **动态预取**：Learned Cache 会从已生成但不在 GPU cache 中的历史 block 里选择高重要性 block 预取回 GPU cache。
- **头部稀疏性**：仿真器为每个 block 生成合成的 `head_activity` 和 `sparse_head` 特征，模拟部分 attention head 激活较低的情况。

## 1. 安装依赖

确认你在虚拟环境里，例如：

```powershell
(KV_Cache) PS D:\PycharmProjects\PythonProject2\KV_Cache>
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

检查依赖：

```powershell
python -c "import numpy, pandas, matplotlib, sklearn; print('ok')"
```

## 2. 运行仿真

默认实验：

```powershell
python kv_cache_sim.py --output-dir my_run_async
```

换随机种子：

```powershell
python kv_cache_sim.py --seed 7 --output-dir my_run_async_seed7
```

改变并发请求数量：

```powershell
python kv_cache_sim.py --num-requests 64 --output-dir my_run_64req
```

运行更大规模实验：

```powershell
python kv_cache_sim.py --extended --output-dir my_run_async_extended
```

运行时看到下面这种提示是正常的：

```text
Running capacity=128 ...
Running capacity=256 ...
Running seq_len=1024 ...
```

## 3. 输出文件

运行后会生成你指定的输出目录，例如 `my_run_async_extended/`：

- `summary.csv`：所有实验结果表。
- `capacity_hit_rate.svg` 和 `capacity_hit_rate.png`：不同 cache 容量下的命中率。
- `capacity_avg_latency.svg` 和 `capacity_avg_latency.png`：不同 cache 容量下的平均访问延迟。
- `seq_len_avg_latency.svg` 和 `seq_len_avg_latency.png`：不同序列长度下的平均访问延迟。
- `locality_mistake_rate.svg` 和 `locality_mistake_rate.png`：不同局部性场景下的误驱逐率。

`summary.csv` 中的关键指标：

- `hit_rate`：缓存命中率。
- `avg_latency`：平均访问延迟。
- `evictions`：驱逐次数。
- `swap_ins`：CPU 到 GPU 的换入次数。
- `prefetches`：Learned Cache 触发的预取次数。
- `mistake_rate`：驱逐后很快又被访问的比例。

## 4. 仿真模型

程序把 KV Cache 抽象成固定大小的 block：

- 一个 block 默认对应 16 个 token 的 KV。
- GPU cache 容量用 block 数表示，例如 256 blocks。
- 多个请求共享同一个 GPU cache。
- 多个请求采用异步突发式推进：每次随机选择一个已到达请求，连续生成 1 到 4 个 block 后再切换请求，避免所有请求机械齐步推进。
- 每一步生成会访问若干历史 block。
- 访问来源分为 recent、prefix 和 random/middle 三类。

五种策略含义：

- FIFO：最早进入 cache 的 block 先驱逐。
- LRU：最久没有被访问的 block 先驱逐。
- LFU：历史访问次数最少的 block 先驱逐。
- Heuristic：根据 recent、prefix、访问频率、head activity 等特征打分。
- Learned：预测重用距离，并结合规则重要性和 head activity 进行驱逐与预取。

## 5. 论文描述示例

本文构建了一个基于 Python 的 KV Cache 管理仿真器。仿真器将 KV Cache 划分为固定大小的 block，并模拟多请求长文本推理过程中的 KV block 访问、GPU cache 命中、CPU 换入、block 驱逐与预取行为。为了体现大模型推理中的时间局部性与稀疏性，仿真器在访问序列生成阶段设置 recent、prefix 和 random/middle 三类访问来源，并为每个 block 生成合成的 attention head activity 特征。

在此基础上，本文实现了 FIFO、LRU、LFU、Heuristic 以及 Learned Cache 策略。Learned Cache 使用轻量级决策树回归模型预测 KV block 的未来重用距离，并结合 recent/prefix 规则先验和 head activity 特征计算 block 重要性。当 GPU cache 满时，系统优先驱逐重要性最低的 block；同时，对于已经生成但不在 GPU cache 中的历史 block，系统根据预测重要性执行保守预取。实验从缓存命中率、平均访问延迟、换入次数、预取次数和误驱逐率等指标进行对比分析。

## 6. 建议实验顺序

先跑默认实验：

```powershell
python kv_cache_sim.py --output-dir my_run_async
```

再改随机种子跑一版，验证结论是否稳定：

```powershell
python kv_cache_sim.py --seed 7 --output-dir my_run_async_seed7
```

如果时间够，再跑扩展版：

```powershell
python kv_cache_sim.py --extended --output-dir my_run_async_extended
```

论文里优先保留三组图：

- 不同 cache 容量下的命中率。
- 不同序列长度下的平均访问延迟。
- 不同访问局部性下的误驱逐率。
