# 学习型 KV Cache 管理策略适用性研究

## 摘要

本文构建异步多请求 KV Cache 仿真器，研究学习型缓存适用性。策略用决策树预测重用距离，结合局部性和头部活跃度驱逐与预取。默认 256 blocks 下，Learned 命中率 0.666、延迟 4.663，优于 LRU 等基线；但容量两端优势减弱，说明其更适合作为传统策略增强。

## 关键词

大语言模型推理；KV Cache；学习型缓存；重用距离；预取

## 1 引言

Transformer 大模型在推理阶段通常采用自回归方式生成文本。每生成一个新 token，模型都需要访问历史 token 对应的 Key 和 Value 矩阵。为了避免重复计算，推理系统会将这些历史 Key 和 Value 缓存在内存中，形成 KV Cache。KV Cache 能显著降低重复计算开销，但也带来显存占用快速增长的问题。

在多轮对话、长文本输入和多请求并发服务场景下，KV Cache 的规模会随序列长度、请求数量、模型层数和 attention head 数量增加而迅速扩大。由于 GPU 显存有限，系统不能无限制地保留所有历史 KV block，因此必须决定哪些 block 保留在 GPU cache 中，哪些 block 被驱逐到低速内存，以及哪些 block 值得提前预取回来。

传统缓存策略通常依赖固定规则。例如 FIFO 按进入缓存的先后顺序驱逐，LRU 按最近访问时间驱逐，LFU 按历史访问频率驱逐。这些策略实现简单，但难以充分利用大模型推理中的结构信息。实际访问模式中，近期 token 往往具有时间局部性，前缀 prompt 或系统提示词可能长期保持重要性，不同 attention head 的活跃程度也可能不同。因此，学习型缓存的思路是：不只依赖固定规则，而是使用轻量模型预测 KV block 未来的重用价值，并据此进行驱逐和预取。

本文围绕“学习型缓存是否适用于大模型推理 KV Cache 管理”这一问题展开研究。本文构建一个 Python 仿真原型，将 KV Cache 抽象为固定大小的 block，模拟异步多请求长文本推理过程中的访问、换入、驱逐与预取行为。在此基础上，本文设计 Learned Cache 策略，使用轻量级决策树回归模型预测 KV block 的未来重用距离，并结合访问局部性和头部活跃度计算 block 重要性。本文重点不是证明 Learned Cache 在所有场景下都优于传统策略，而是分析其在哪些条件下有效、在哪些条件下优势减弱。

## 2 背景与相关工作

### 2.1 Transformer 推理与 KV Cache

Transformer 模型由 Vaswani 等人提出，其核心是多头自注意力机制[1]。在自回归推理中，当前 token 的注意力计算需要依赖此前所有 token 的 Key 和 Value。如果每一步都重新计算完整历史序列，推理成本会随上下文长度显著增加。因此，现代大语言模型推理系统通常缓存历史 Key 和 Value，只为新 token 追加新的 KV 状态。

KV Cache 的优势是避免重复计算，代价是显存占用随上下文长度和并发请求数线性增长。在长上下文和多用户并发服务中，KV Cache 往往成为限制吞吐量和请求规模的重要因素。因此，KV Cache 不仅是模型推理中的中间状态，也是推理系统需要重点管理的内存资源。已有工作从不同角度降低推理内存或带宽压力，例如 Multi-Query Attention 和 Grouped-Query Attention 通过减少 KV head 数量降低解码阶段的 KV 读写开销[2-3]，FlashAttention 系列从 IO 感知角度优化 attention 计算[8,15]，大规模 Transformer 推理系统研究也强调了 KV Cache 对长序列推理效率的影响[4]。

### 2.2 KV Cache 管理策略

在推理系统层面，Orca 通过 iteration-level scheduling 和选择性 batching 改进生成式模型服务效率[5]，FlexGen 则利用 GPU、CPU 和磁盘之间的分层存储与调度，在有限 GPU 资源下支持大模型推理[6]。vLLM 提出的 PagedAttention 将 KV Cache 管理与操作系统分页思想结合，把连续 KV Cache 拆分成块进行管理，从而降低显存碎片并支持灵活共享[7]。这些系统说明，KV Cache 的内存布局、调度和替换策略都会直接影响大语言模型服务效率。

除了内存布局优化，KV Cache 替换策略同样重要。当 GPU cache 容量不足时，系统必须选择部分 block 驱逐。传统 FIFO、LRU、LFU 等策略没有直接理解大模型推理中的访问结构，可能无法在复杂长上下文中准确判断 block 未来价值。H2O 从注意力分布出发，认为少量 heavy-hitter token 对生成质量具有更大影响，应优先保留近期 token 与重要 token[9]。StreamingLLM 指出，初始 token 可能具有 attention sink 作用，不应简单按照滑动窗口全部丢弃[10]。Scissorhands、SnapKV、FastGen 和 PyramidKV 等工作进一步表明，KV Cache 中不同 token、不同层或不同 attention head 的重要性存在差异，可以通过注意力分布、观察窗口或模型 profiling 进行压缩和保留决策[11-14]。

这些工作表明，KV Cache 管理不能只看访问时间，还应考虑 token 位置、历史重要性和注意力结构。与此同时，学习型索引和学习型缓存替换研究说明，轻量模型可以用于预测访问位置、工作负载类型或缓存块价值，从而辅助传统系统策略[16-17]。本文不直接修改真实推理框架，而是在课程项目范围内构建仿真器，验证一种轻量学习型驱逐与预取策略的适用性。

## 3 方法设计

### 3.1 KV Cache 块模型

本文将 KV Cache 按 block 管理，而不是按单个 token 管理。默认设置下，一个 block 对应 16 个 token 的 KV 状态。对于长度为 2048 token 的请求，一个请求会产生 128 个 KV block。仿真器设置 32 个并发请求共享同一个 GPU cache，默认 GPU cache 容量为 256 blocks。

采用 block 粒度有两个原因。第一，真实推理系统通常不会以单 token 为单位进行内存管理，块级管理更接近 PagedAttention 等系统设计[7]。第二，block 粒度能降低仿真复杂度，使实验重点放在替换策略和预取策略本身。

### 3.2 异步访问模式建模

早期同步推进的多请求 trace 会造成不自然的访问偏置：所有请求在同一个生成进度上齐步推进，容易让 LRU 表现异常。因此，本文最终采用异步突发式请求调度。具体而言，仿真器每次随机选择一个已到达请求，连续推进 1 到 4 个 block 后再切换到其他请求。该设置更接近实际服务中多个请求进度不完全一致的情况。

每一步生成时，历史 block 的访问来源分为三类。第一类是 recent block，即当前生成位置附近的近期 block，默认比例为 60%，用于模拟近期上下文被频繁访问的时间局部性。第二类是 prefix block，默认比例为 25%，用于模拟系统提示词、任务说明和问题开头等前缀信息的长期影响。第三类是 random/middle block，默认比例为 15%，用于模拟长距离依赖和稀疏历史访问。上述比例对应实验中的 medium locality profile；扩展实验进一步设置 strong 和 weak 两种局部性，用于检验策略对不同访问模式的适应性。

为了体现题目中提到的头部激活稀疏性，仿真器为每个 block 生成合成的 head activity 分数。该分数表示该 block 对应 attention head 的平均活跃程度。访问采样时，head activity 较高的 block 更可能被访问；在替换决策中，head activity 较低的 block 重要性相对降低。

### 3.3 特征设计

Learned Cache 使用以下特征描述每个 KV block 的状态。

| 特征 | 含义 | 对应信息 |
| --- | --- | --- |
| last_gap | 距离上次访问的间隔 | 时间局部性 |
| access_count | 历史访问次数 | 频度信息 |
| resident_age | 在缓存中的驻留时长 | 老化信息 |
| distance_to_tail | 距当前生成位置的块距离 | 位置信息 |
| position_ratio | block 在当前上下文中的相对位置 | 位置信息 |
| prefix | 是否属于前缀 block | 前缀先验 |
| recent | 是否位于近期窗口 | 近期先验 |
| head_activity | 合成 attention head 活跃度 | 稀疏性 |
| sparse_head | 是否为低活跃 head block | 稀疏性 |

这些特征既包含传统缓存策略常用的时间和频度信息，也包含更贴近大模型推理场景的位置先验和头部稀疏性信息。

### 3.4 重用距离预测

题目要求预测 KV Cache block 在未来的重用距离。本文将重用距离定义为：从当前时刻开始，到该 block 下一次被访问之间的访问步数。如果某个 block 在观察窗口内不再被访问，则将其重用距离设为较大的截断值。

本文使用 `DecisionTreeRegressor` 作为轻量级预测模型。选择决策树主要基于三点考虑。第一，决策树推理开销较小，适合在线缓存决策。第二，决策树对特征尺度不敏感，便于处理访问间隔、频次、位置比例等异构特征。第三，决策树具有较强可解释性，符合课程项目对轻量模型的要求。实验中决策树最大深度设为 8，叶子节点最小样本数设为 12，训练样本从合成访问 trace 中抽取，最多使用 24000 条样本。

### 3.5 驱逐与预取决策

Learned Cache 首先利用决策树预测候选 block 的未来重用距离。预测距离越短，说明该 block 越可能在近期被再次访问，应该保留或预取。为了增强稳定性，本文进一步融合启发式重要性和 head activity，得到最终重要性分数：

```text
importance = 0.55 * distance_score
           + 0.30 * heuristic_score
           + 0.15 * head_activity
```

其中 `distance_score` 由预测重用距离转换而来，预测距离越短，该分数越高；`heuristic_score` 综合 recent、prefix、访问频次等先验；`head_activity` 表示该 block 对应 attention head 的活跃度。

上述 0.55、0.30 和 0.15 的权重为经验设定，用于平衡模型预测、规则先验和稀疏性信号。本文重点验证学习型缓存策略的可行性，尚未对该融合权重进行系统化超参数搜索。

当 GPU cache 已满且需要放入新 block 时，Learned Cache 选择 importance 最低的 block 驱逐。对于当前请求中已经生成但不在 GPU cache 中的历史 block，系统会计算其 importance，若分数超过阈值并且优于当前 cache 中最弱 block，则执行预取。预取会产生较小的额外延迟，但可以减少后续按需换入带来的高代价 miss。

## 4 仿真实验与结果分析

### 4.1 实验设置

本文实现了一个 Python KV Cache 管理仿真器。仿真器包括访问序列生成、缓存状态维护、替换策略执行、预取决策和指标统计等模块。对比策略包括 FIFO、LRU、LFU、Heuristic 和 Learned Cache。

默认实验参数如下：并发请求数为 32，序列长度为 2048 token，block size 为 16 token，GPU cache 容量为 256 blocks。请求调度采用异步突发模式：每次随机选择一个已到达请求，连续推进 1 到 4 个 block 后再切换请求。延迟参数设置为：GPU cache 命中代价为 1，CPU 换入代价为 10，驱逐代价为 2，预取代价为 0.2。评价指标包括缓存命中率、平均访问延迟、驱逐次数、按需换入次数、预取次数和误驱逐率。

本文使用三组实验结果：`my_run_async` 作为默认主实验，`my_run_async_seed7` 用于随机种子稳定性验证，`my_run_async_extended` 用于容量、序列长度和访问局部性扩展实验。

### 4.2 默认场景结果

默认场景下，各策略结果如下。

表 1 默认场景下不同缓存策略的对比结果

| 策略 | 命中率 | 平均延迟 | 驱逐次数 | 按需换入 | 预取次数 | 误驱逐率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FIFO | 0.624 | 5.120 | 14940 | 15196 | 0 | 0.092 |
| LRU | 0.632 | 5.036 | 14630 | 14886 | 0 | 0.081 |
| LFU | 0.342 | 8.221 | 26340 | 26596 | 0 | 0.441 |
| Heuristic | 0.615 | 5.221 | 15310 | 15566 | 0 | 0.254 |
| Learned | 0.666 | 4.663 | 13303 | 13506 | 53 | 0.169 |

结果显示，在异步多请求场景下，LRU 已经是较强基线，其命中率达到 0.632，平均访问延迟为 5.036，明显优于 FIFO 和 LFU。这说明在更自然的异步访问模式下，最近性信息仍然具有较强参考价值，传统策略并非完全失效。

Learned Cache 在默认容量 256 blocks 下取得最高命中率 0.666 和最低平均延迟 4.663。与 LRU 相比，Learned Cache 命中率提高 3.4 个百分点，平均访问延迟降低 0.373，按需换入次数由 14886 降至 13506；与 Heuristic 相比，命中率提高 5.1 个百分点，平均访问延迟降低 0.558，按需换入次数减少 2060 次。该结果说明，重用距离预测能够在最近性和手工规则之外提供额外判断信息。

同时需要注意，Learned Cache 的预取次数仅为 53 次，说明该策略在异步 workload 下并不是依靠大量预取取得收益，而主要来自更合理的驱逐决策。预取在本实验中更像辅助机制：当模型预测某些不在 GPU cache 中的历史 block 具有较高近期重用价值时，才会触发保守预取。

### 4.3 随机种子稳定性

为了检查结果是否由单次随机 trace 偶然造成，本文将随机种子改为 7 进行重复实验。结果显示，在 seed=7 下，LRU 命中率为 0.629，平均延迟为 5.070；Heuristic 命中率为 0.611，平均延迟为 5.266；Learned Cache 命中率为 0.672，平均延迟为 4.597。Learned Cache 仍然优于 LRU 和 Heuristic。

该结果说明，在不同随机访问序列下，基于重用距离预测的策略仍能保持一定优势。与默认随机种子相比，Learned Cache 的命中率从 0.666 变化为 0.672，平均延迟从 4.663 变化为 4.597，趋势基本一致。seed=7 下 Learned Cache 的预取次数为 75 次，仍然较少，说明异步多请求场景中的预取机制整体偏保守，主要收益依旧来自驱逐决策。

### 4.4 扩展实验

扩展实验从 cache 容量、序列长度和访问局部性三个维度进行分析。不同 cache 容量实验用于观察 GPU cache 资源变化对策略表现的影响；不同序列长度实验用于模拟长上下文场景；不同访问局部性实验用于检验策略在强局部性、中等局部性和弱局部性访问模式下的稳定性。表 2 展示了不同容量下的命中率，图 1 至图 4 进一步展示了三个维度的趋势。

表 2 不同 cache 容量下的命中率

| capacity | FIFO | LRU | LFU | Heuristic | Learned |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.482 | 0.485 | 0.173 | 0.220 | 0.432 |
| 128 | 0.538 | 0.544 | 0.302 | 0.350 | 0.522 |
| 256 | 0.624 | 0.632 | 0.342 | 0.615 | 0.666 |
| 512 | 0.750 | 0.761 | 0.405 | 0.827 | 0.764 |

![图1 不同 cache 容量下的命中率](D:/PycharmProjects/PythonProject2/KV_Cache/my_run_async_extended/capacity_hit_rate.svg)

![图2 不同 cache 容量下的平均访问延迟](D:/PycharmProjects/PythonProject2/KV_Cache/my_run_async_extended/capacity_avg_latency.svg)

从容量实验可以看出，随着 cache 容量增大，各策略命中率整体上升，平均访问延迟下降。Learned Cache 的命中率也随容量增加而提高：在 64、128、256 和 512 blocks 下分别为 0.432、0.522、0.666 和 0.764。这说明学习型策略能够从更大的缓存空间中受益，但其相对优势并非在所有容量下都成立。

具体来看，在 capacity 为 64 和 128 blocks 时，LRU 的命中率分别为 0.485 和 0.544，高于 Learned Cache 的 0.432 和 0.522。此时 cache 极度紧张，模型预测和预取候选的误差更容易放大，简单最近性策略反而更稳。capacity 为 256 blocks 时，Learned Cache 命中率达到 0.666，高于 LRU 的 0.632 和 Heuristic 的 0.615，说明中等容量下模型预测能够提供额外收益。当 capacity 增大至 512 blocks 时，Heuristic 命中率为 0.827，高于 Learned Cache 的 0.764，表明当缓存空间较充足时，规则策略已经能够覆盖大部分工作集，学习型策略的边际收益下降。

这一结果说明，Learned Cache 的适用性具有容量敏感性。它不是 LRU 或 Heuristic 的无条件替代，而更适合作为中等资源约束下的学习型增强。未来策略可以根据 cache 容量和当前负载动态调整预取阈值与重要性权重。

![图3 不同序列长度下的平均访问延迟](D:/PycharmProjects/PythonProject2/KV_Cache/my_run_async_extended/seq_len_avg_latency.svg)

在序列长度实验中，Learned Cache 在 1024、2048 和 4096 token 场景下均取得较低平均延迟；但在 512 token 的较短上下文中，LRU 的表现更接近甚至略优。这说明当上下文较短时，最近性已经足以描述大部分访问模式；随着序列长度增加，历史 block 的重用关系更复杂，重用距离预测的价值才逐渐体现。

不同访问局部性下 LRU、Heuristic 和 Learned Cache 的命中率如下。

表 3 不同访问局部性下的命中率

| locality | LRU | Heuristic | Learned |
| --- | ---: | ---: | ---: |
| strong | 0.717 | 0.681 | 0.754 |
| medium | 0.632 | 0.615 | 0.666 |
| weak | 0.492 | 0.477 | 0.519 |

![图4 不同局部性下的误驱逐率](D:/PycharmProjects/PythonProject2/KV_Cache/my_run_async_extended/locality_mistake_rate.svg)

在不同局部性设置下，Learned Cache 在 strong、medium 和 weak 三类访问模式中均取得最高命中率，分别为 0.754、0.666 和 0.519。尤其在 weak locality 场景下，访问更加分散，LRU 和 Heuristic 的命中率分别为 0.492 和 0.477，而 Learned Cache 仍达到 0.519。这说明当访问局部性减弱、单一最近性规则不够充分时，多特征学习策略更容易体现优势。

## 5 局限性与未来工作

本文实验仍存在一定局限。第一，访问 trace 由仿真器合成生成，并非真实大语言模型推理 trace，因此只能验证策略趋势，不能完全代表真实部署环境。第二，head activity 是合成特征，用于模拟 attention head 稀疏性，并未直接来自真实模型的 attention 分布。第三，预取机制采用保守阈值和固定预取预算，尚未考虑 GPU 带宽竞争、异步拷贝调度等真实系统因素。第四，importance 中不同权重采用经验设置，尚未进行系统化超参数搜索。第五，当前策略没有根据 cache 容量动态调整预取强度，在极小或较大容量下可能不如 LRU 或 Heuristic。第六，本文只比较了有限数量的替换策略，未来可加入更多 KV Cache 压缩、量化或分层存储策略。

后续工作可以从三个方向展开。首先，将仿真器接入真实 vLLM 或其他推理框架，采集真实请求 trace 和 attention 分数。其次，引入在线学习机制，使模型能够根据当前请求分布持续更新。最后，将预取代价与 GPU 带宽、CPU-GPU 传输延迟联合建模，使仿真更接近实际部署环境。

## 6 结论

本文围绕大语言模型推理中的 KV Cache 管理问题，构建了异步多请求块级缓存管理仿真器，并提出基于重用距离预测的 Learned Cache 策略。该策略使用轻量级决策树回归模型预测 KV block 的未来重用距离，结合时间局部性、前缀先验和头部活跃度计算 block 重要性，从而动态执行驱逐与预取。

实验表明，Learned Cache 在默认场景下优于 LRU 和 Heuristic，但优势具有容量敏感性：在 cache 极度紧张或较为充裕时，传统策略仍可能表现更好。预取机制在异步多请求场景下触发偏少，主要收益来自驱逐决策。因此，学习型 KV Cache 管理更适合作为传统策略的增强机制，而不是简单替代。未来若能结合真实推理 trace、在线学习和容量感知预取调节，该方向有望为生产环境中的 KV Cache 管理提供更稳健的轻量级学习辅助。

## 参考文献

[1] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need[C]//Advances in Neural Information Processing Systems. 2017, 30: 5998-6008.

[2] Shazeer N. Fast Transformer Decoding: One Write-Head is All You Need[R/OL]. arXiv:1911.02150, 2019[2026-06-04]. https://arxiv.org/abs/1911.02150.

[3] Ainslie J, Lee-Thorp J, de Jong M, et al. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints[A/OL]//Bouamor H, Pino J, Bali K, eds. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing[C]. Singapore: Association for Computational Linguistics, 2023: 4895-4901[2026-06-04]. https://aclanthology.org/2023.emnlp-main.298/.

[4] Pope R, Douglas S, Chowdhery A, et al. Efficiently Scaling Transformer Inference[A/OL]//Song D, Carbin M, Chen T, eds. Proceedings of Machine Learning and Systems[C]. 2023, 5: 606-624[2026-06-04]. https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html.

[5] Yu G I, Jeong J S, Kim G W, et al. Orca: A Distributed Serving System for Transformer-Based Generative Models[A]//Proceedings of the 16th USENIX Symposium on Operating Systems Design and Implementation[C]. Berkeley: USENIX Association, 2022: 521-538.

[6] Sheng Y, Zheng L, Yuan B, et al. FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU[A/OL]//Krause A, Brunskill E, Cho K, et al, eds. Proceedings of the 40th International Conference on Machine Learning[C]. PMLR, 2023, 202: 31094-31116[2026-06-04]. https://proceedings.mlr.press/v202/sheng23a.html.

[7] Kwon W, Li Z, Zhuang S, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention[A/OL]//Proceedings of the 29th ACM Symposium on Operating Systems Principles[C]. New York: ACM, 2023: 611-626[2026-06-04]. https://doi.org/10.1145/3600006.3613165.

[8] Dao T, Fu D Y, Ermon S, et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness[A/OL]//Advances in Neural Information Processing Systems[C]. 2022, 35: 16344-16359[2026-06-04]. https://arxiv.org/abs/2205.14135.

[9] Zhang Z, Sheng Y, Zhou T, et al. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models[A/OL]//Advances in Neural Information Processing Systems[C]. 2023, 36[2026-06-04]. https://proceedings.neurips.cc/paper_files/paper/2023/hash/6ceefa7b15572587b78ecfcebb2827f8-Abstract-Conference.html.

[10] Xiao G, Tian Y, Chen B, et al. Efficient Streaming Language Models with Attention Sinks[A/OL]//International Conference on Learning Representations[C]. 2024[2026-06-04]. https://openreview.net/forum?id=NG7sS51zVF.

[11] Liu Z, Desai A, Liao F, et al. Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time[R/OL]. arXiv:2305.17118, 2023[2026-06-04]. https://arxiv.org/abs/2305.17118.

[12] Li Y, Huang Y, Yang B, et al. SnapKV: LLM Knows What You are Looking for Before Generation[R/OL]. arXiv:2404.14469, 2024[2026-06-04]. https://arxiv.org/abs/2404.14469.

[13] Ge S, Zhang Y, Liu L, et al. Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs[A/OL]//International Conference on Learning Representations[C]. 2024[2026-06-04]. https://openreview.net/pdf?id=uNrFpDPMyo.

[14] Cai Z, Zhang Y, Gao B, et al. PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling[R/OL]. arXiv:2406.02069, 2024[2026-06-04]. https://arxiv.org/abs/2406.02069.

[15] Dao T. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning[A/OL]//International Conference on Learning Representations[C]. 2024[2026-06-04]. https://openreview.net/forum?id=mZn2Xyh9Ec.

[16] Kraska T, Beutel A, Chi E H, et al. The Case for Learned Index Structures[A]//Proceedings of the ACM SIGMOD International Conference on Management of Data[C]. New York: ACM, 2018: 489-504.

[17] Vietri G, Rodriguez L V, Martinez W A, et al. Driving Cache Replacement with ML-based LeCaR[A]//Proceedings of the 10th USENIX Workshop on Hot Topics in Storage and File Systems[C]. Berkeley: USENIX Association, 2018.
