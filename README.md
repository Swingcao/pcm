# PCM: Proactive Cognitive Memory System

<div align="center">

**基于惊奇度驱动的自演化智能体记忆框架**

*A Surprise-Driven Self-Evolving Memory Framework for Intelligent Agents*

</div>

---

## 目录

- [项目概述](#项目概述)
- [核心创新](#核心创新)
- [系统架构](#系统架构)
- [三层架构详解](#三层架构详解)
  - [Layer 1: 感知与工作记忆层](#layer-1-感知与工作记忆层)
  - [Layer 2: 概率世界模型层](#layer-2-概率世界模型层)
  - [Layer 3: 认知演化引擎](#layer-3-认知演化引擎)
- [数学基础](#数学基础)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [评估框架](#评估框架)
- [项目结构](#项目结构)

---

## 项目概述

PCM (Proactive Cognitive Memory) 是一个具备**主动感知**和**自主演化**能力的智能体记忆系统。与传统被动式记忆不同，PCM 能够：

- **主动检测异常**：通过惊奇度 (Surprisal) 监测识别与已有知识的冲突
- **自主更新信念**：基于贝叶斯推理动态调整知识置信度
- **隐式验证假设**：通过未来观察验证推测，而非显式询问用户
- **区分不确定性类型**：分离认知不确定性（可减少）和偶然不确定性（不可减少）

---

## 核心创新

### 1. 惊奇度驱动的系统演化

```
传统方法: 用户输入 → 检索 → 存储 (被动)
PCM方法:  用户输入 → 惊奇度评估 → 分级响应 → 自主演化 (主动)
```

### 2. 有效惊奇度公式

```
S_eff(u_t) = S_raw(u_t) × (1 - λ × H(C_t))
```

- `S_raw`: 原始惊奇度 (NLL)
- `H(C_t)`: 检索熵（检索质量指标）
- `λ`: 熵权重系数

**关键洞察**：高 NLL 可能意味着"冲突"或"无知"，通过检索熵区分这两种情况。

### 3. 三级代理响应机制

| 惊奇度级别 | 触发代理 | 操作 |
|-----------|---------|------|
| 高 (S_eff > θ_high) | Correction Agent | 衰减冲突节点，创建新事实 |
| 中 (θ_low < S_eff ≤ θ_high) | Profiling Agent | 生成假设节点 |
| 低 (S_eff ≤ θ_low) | Maintenance Agent | 强化现有知识 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PCM System Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Layer 1: Perception Layer                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
│  │  │  Sliding    │  │   Intent    │  │     Surprise Monitor    │  │   │
│  │  │  Context    │→ │   Router    │→ │  ┌─────────────────┐    │  │   │
│  │  │   Queue     │  │             │  │  │ S_eff = S_raw × │    │  │   │
│  │  │  Q_{t-1}    │  │ P(I|u,Q)    │  │  │ (1 - λ·H(C))   │    │  │   │
│  │  └─────────────┘  └─────────────┘  │  └─────────────────┘    │  │   │
│  │        ↓                ↓          └───────────┬─────────────┘  │   │
│  └────────┼────────────────┼──────────────────────┼────────────────┘   │
│           │                │                      │                     │
│           ↓                ↓                      ↓                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                Layer 2: Probabilistic World Model                │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │              Weighted Knowledge Graph G_t               │    │   │
│  │  │   ┌─────┐      ┌─────┐      ┌─────┐      ┌─────┐       │    │   │
│  │  │   │ v_1 │──w──→│ v_2 │──w──→│ v_3 │──w──→│ v_4 │       │    │   │
│  │  │   │ 0.8 │      │ 0.6 │      │ 0.4 │      │ 0.9 │       │    │   │
│  │  │   └─────┘      └─────┘      └─────┘      └─────┘       │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  │  ┌──────────────────┐  ┌──────────────────────────────────┐    │   │
│  │  │   NetworkX Graph │  │        ChromaDB Vector Index     │    │   │
│  │  │   (Structure)    │  │         (Semantic Search)        │    │   │
│  │  └──────────────────┘  └──────────────────────────────────┘    │   │
│  │                              ↑                                  │   │
│  │           Intent-Masked Retrieval: Score = sim × P(d|I) × w    │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
│                                 │                                       │
│                                 ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                Layer 3: Cognitive Evolution Engine               │   │
│  │                                                                  │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │                  Surprise Dispatcher                      │   │   │
│  │  │           S_eff → Route to appropriate agent              │   │   │
│  │  └──────────────────────┬───────────────────────────────────┘   │   │
│  │                         │                                        │   │
│  │         ┌───────────────┼───────────────┐                       │   │
│  │         ↓               ↓               ↓                       │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │   │
│  │  │ Correction  │ │  Profiling  │ │ Maintenance │               │   │
│  │  │   Agent     │ │   Agent     │ │   Agent     │               │   │
│  │  │             │ │             │ │             │               │   │
│  │  │ HIGH surp.  │ │ MEDIUM surp.│ │ LOW surp.   │               │   │
│  │  │ w *= e^{-β} │ │ Create hypo │ │ w += η(1-w) │               │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 三层架构详解

### Layer 1: 感知与工作记忆层

**职责**：实时交互处理、意图分类、惊奇度计算

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Layer 1: Perception Layer                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Input (u_t)                                                        │
│       │                                                                  │
│       ↓                                                                  │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │              SlidingContextQueue (Working Memory)              │     │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                │     │
│  │  │u_{-4}│→│r_{-4}│→│u_{-2}│→│r_{-2}│→│ u_t  │  ← 新输入      │     │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘                │     │
│  │                                         │                     │     │
│  │  当 tokens > MAX_TOKENS 时:             │                     │     │
│  │  ┌──────┐ ┌──────┐                      │                     │     │
│  │  │u_{-4}│ │r_{-4}│ → 驱逐到 L3 处理      │                     │     │
│  │  └──────┘ └──────┘                      │                     │     │
│  └─────────────────────────────────────────┼────────────────────────┘     │
│                                            │                             │
│       ┌────────────────────────────────────┘                             │
│       │                                                                  │
│       ↓                                                                  │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                      IntentRouter                              │     │
│  │                                                                │     │
│  │   P(I_t | u_t, Q_{t-1}) = Softmax(f_θ(u_t, Q_{t-1}))          │     │
│  │                                                                │     │
│  │   Domains: [Coding, Academic, Personal, Casual, Professional]  │     │
│  │                                                                │     │
│  │   Output: Intent {                                             │     │
│  │     label: "Coding",                                           │     │
│  │     confidence: 0.85,                                          │     │
│  │     distribution: {Coding: 0.85, Academic: 0.10, ...}          │     │
│  │   }                                                            │     │
│  └────────────────────────────────────────┬───────────────────────┘     │
│                                           │                              │
│       ┌───────────────────────────────────┘                              │
│       │                                                                  │
│       ↓                                                                  │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                    SurpriseMonitor                             │     │
│  │                                                                │     │
│  │   Step 1: Raw Surprisal (NLL from local LM)                    │     │
│  │   ┌────────────────────────────────────────────┐               │     │
│  │   │ S_raw = -log P_LM(u_t | C_retrieved)       │               │     │
│  │   └────────────────────────────────────────────┘               │     │
│  │                                                                │     │
│  │   Step 2: Retrieval Entropy                                    │     │
│  │   ┌────────────────────────────────────────────┐               │     │
│  │   │ H(C_t) = -Σ ŝ(ε) × log(ŝ(ε))              │               │     │
│  │   │ (normalized retrieval scores)              │               │     │
│  │   └────────────────────────────────────────────┘               │     │
│  │                                                                │     │
│  │   Step 3: Effective Surprisal                                  │     │
│  │   ┌────────────────────────────────────────────┐               │     │
│  │   │ S_eff = S_raw × (1 - λ × H(C_t))          │               │     │
│  │   │                                            │               │     │
│  │   │ 高熵(检索不确定) → 折扣惊奇度 (可能是无知)   │               │     │
│  │   │ 低熵(检索确定)   → 保留惊奇度 (可能是冲突)   │               │     │
│  │   └────────────────────────────────────────────┘               │     │
│  │                                                                │     │
│  │   Output: SurprisalPacket {                                    │     │
│  │     raw_score: 4.2,                                            │     │
│  │     effective_score: 3.1,                                      │     │
│  │     retrieval_entropy: 0.26,                                   │     │
│  │     level: "high" | "medium" | "low"                           │     │
│  │   }                                                            │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 核心组件

| 组件 | 文件 | 功能 |
|-----|------|------|
| `SlidingContextQueue` | `layer1_perception.py` | 滑动窗口工作记忆，自动驱逐旧内容 |
| `IntentRouter` | `layer1_perception.py` | LLM-based 意图分类器 |
| `SurpriseMonitor` | `layer1_perception.py` | 惊奇度计算与级别判定 |

#### 关键代码

```python
# 感知层处理流程
async def process_input(self, user_input, retrieved_nodes, retrieval_scores):
    # 1. 获取当前上下文
    context = self.context_queue.get_context(max_turns=5)

    # 2. 添加到队列（可能触发驱逐）
    evicted = self.context_queue.add("user", user_input)

    # 3. 意图分类
    intent = await self.intent_router.classify(user_input, context)

    # 4. 惊奇度计算
    surprisal_packet = self.surprise_monitor.calculate_surprisal(
        user_input, retrieved_nodes, retrieval_scores
    )

    return evicted, intent, surprisal_packet
```

---

### Layer 2: 概率世界模型层

**职责**：知识存储、语义检索、置信度管理

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Layer 2: Probabilistic World Model                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                  Weighted Knowledge Graph G_t                  │     │
│  │                                                                │     │
│  │   Node Types:                                                  │     │
│  │   ● ENTITY (实体)      ○ ATTRIBUTE (属性)                      │     │
│  │   ◆ HYPOTHESIS (假设)  ★ FACT (事实)                           │     │
│  │                                                                │     │
│  │         "Python"                    "ML Research"              │     │
│  │         ●(0.9)                      ●(0.7)                     │     │
│  │            │ expert_in                 │ interested_in         │     │
│  │            │ w=0.85                    │ w=0.6                  │     │
│  │            ↓                           ↓                       │     │
│  │      ┌──────────────────────────────────────┐                  │     │
│  │      │           "User Profile"             │                  │     │
│  │      │              ★(0.8)                  │                  │     │
│  │      └──────────────────────────────────────┘                  │     │
│  │            │                           │                       │     │
│  │            │ prefers                   │ studies               │     │
│  │            │ w=0.7                     │ w=0.5                  │     │
│  │            ↓                           ↓                       │     │
│  │    "Backend Dev"              "Transformers"                   │     │
│  │     ○(0.75)                    ◆(0.4)                          │     │
│  │                               (hypothesis)                     │     │
│  │                                                                │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌─────────────────────────┐  ┌─────────────────────────────────┐      │
│  │     NetworkX DiGraph    │  │      ChromaDB Collection        │      │
│  │  ┌───────────────────┐  │  │  ┌───────────────────────────┐  │      │
│  │  │ • Node attributes │  │  │  │ • Vector embeddings       │  │      │
│  │  │ • Edge weights    │  │  │  │ • Semantic similarity     │  │      │
│  │  │ • Graph traversal │  │  │  │ • Metadata filtering      │  │      │
│  │  │ • Persistence     │  │  │  │ • Cosine distance search  │  │      │
│  │  └───────────────────┘  │  │  └───────────────────────────┘  │      │
│  └─────────────────────────┘  └─────────────────────────────────┘      │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                  Intent-Masked Retrieval                       │     │
│  │                                                                │     │
│  │   Score(ε_k) = sim(emb(u_t), emb(ε_k)) × P(d(ε_k)|I_t) × w_k  │     │
│  │                    ↑                       ↑              ↑    │     │
│  │              语义相似度              意图相关性      置信权重   │     │
│  │                                                                │     │
│  │   Example:                                                     │     │
│  │   Query: "How to use PyTorch?"                                 │     │
│  │   Intent: Coding (0.9)                                         │     │
│  │                                                                │     │
│  │   Node "Python expert" (domain: Coding):                       │     │
│  │   Score = 0.8 × 0.9 × 0.85 = 0.612                            │     │
│  │                                                                │     │
│  │   Node "Likes hiking" (domain: Personal):                      │     │
│  │   Score = 0.3 × 0.1 × 0.7 = 0.021  (filtered by intent)       │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                    Soft Weight Updates                         │     │
│  │                                                                │     │
│  │   Reinforce (low surprise):                                    │     │
│  │   w_{t+1} = w_t + η × (1 - w_t)    → 渐近趋向 1.0              │     │
│  │                                                                │     │
│  │   Decay (high surprise):                                       │     │
│  │   w_{t+1} = w_t × e^{-β × S_eff}  → 指数衰减                   │     │
│  │                                                                │     │
│  │   注意：永不删除节点，只调整权重 (软更新策略)                    │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 核心组件

| 组件 | 功能 | 数据结构 |
|-----|------|---------|
| `MemoryNode` | 知识节点 | id, content, type, domain, weight |
| `MemoryEdge` | 关系边 | source, target, relation, weight |
| `WeightedKnowledgeGraph` | 图数据库 | NetworkX + ChromaDB |

#### 关键代码

```python
# 意图掩码检索
def retrieve(self, query, intent, top_k):
    # 1. 获取查询嵌入
    query_embedding = self.embedding_model.encode(query)

    # 2. ChromaDB 向量搜索
    results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k*3)

    # 3. 计算组合分数
    for node_id, distance, metadata in results:
        similarity = 1.0 - distance
        node_domain = metadata["domain"]
        intent_relevance = intent.distribution.get(node_domain, 0.1)
        weight = metadata["weight"]

        # 组合分数：相似度 × 意图相关性 × 置信权重
        score = similarity * intent_relevance * weight

    return sorted_nodes, scores
```

---

### Layer 3: 认知演化引擎

**职责**：基于惊奇度分级处理，自主更新世界模型

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Layer 3: Cognitive Evolution Engine                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                         SurprisalPacket                                  │
│                              │                                           │
│                              ↓                                           │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                    Surprise Dispatcher                         │     │
│  │                                                                │     │
│  │         S_eff > θ_high (3.0)?  ──Yes──→  Correction Agent     │     │
│  │              │                                                 │     │
│  │             No                                                 │     │
│  │              ↓                                                 │     │
│  │         S_eff > θ_low (1.0)?   ──Yes──→  Profiling Agent      │     │
│  │              │                                                 │     │
│  │             No                                                 │     │
│  │              ↓                                                 │     │
│  │         Maintenance Agent                                      │     │
│  └─────────────┬─────────────────────────────────────────────────┘     │
│                │                                                         │
│    ┌───────────┼───────────────────┬───────────────────┐                │
│    ↓           ↓                   ↓                   ↓                │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Correction Agent (高惊奇)                    │    │
│  │                                                                 │    │
│  │  触发条件: S_eff > 3.0 (严重冲突)                               │    │
│  │                                                                 │    │
│  │  场景示例:                                                      │    │
│  │  已有知识: "用户喜欢 Python 后端开发"                           │    │
│  │  新输入:   "我决定完全放弃编程，只做 AI 研究"                    │    │
│  │                                                                 │    │
│  │  处理流程:                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────┐  │    │
│  │  │ 1. 诊断冲突                                              │  │    │
│  │  │    LLM 分析: "用户兴趣发生重大转变"                       │  │    │
│  │  │                                                          │  │    │
│  │  │ 2. 衰减旧知识                                            │  │    │
│  │  │    "Python后端" 节点: w = 0.8 × e^{-0.3×3.5} = 0.28     │  │    │
│  │  │                                                          │  │    │
│  │  │ 3. 创建新事实                                            │  │    │
│  │  │    新节点: "Focus on AI research" (w=0.8, type=FACT)     │  │    │
│  │  │    新边: "supersedes" → "Python后端"                     │  │    │
│  │  └──────────────────────────────────────────────────────────┘  │    │
│  │                                                                 │    │
│  │  权重更新公式: w_{t+1} = w_t × e^{-β × S_eff}                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Profiling Agent (中惊奇)                     │    │
│  │                                                                 │    │
│  │  触发条件: 1.0 < S_eff ≤ 3.0 (新颖但不冲突)                    │    │
│  │                                                                 │    │
│  │  场景示例:                                                      │    │
│  │  已有知识: "用户是 Python 开发者"                               │    │
│  │  新输入:   "我最近在学习 Transformer 架构"                      │    │
│  │                                                                 │    │
│  │  处理流程:                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────┐  │    │
│  │  │ 1. 内部推理 (Internal Monologue)                         │  │    │
│  │  │    "用户是Python开发者，现在学习Transformer..."           │  │    │
│  │  │    "可能正在转向 AI/ML 领域..."                          │  │    │
│  │  │                                                          │  │    │
│  │  │ 2. 生成假设                                              │  │    │
│  │  │    HypothesisNode {                                      │  │    │
│  │  │      content: "用户可能正在转型做 AI 研究",               │  │    │
│  │  │      weight: σ(S_eff) ∈ [0.3, 0.5],  // 初始权重低       │  │    │
│  │  │      type: HYPOTHESIS                                    │  │    │
│  │  │    }                                                     │  │    │
│  │  │                                                          │  │    │
│  │  │ 3. 等待隐式验证                                          │  │    │
│  │  │    未来观察如果支持该假设 → 权重增加                      │  │    │
│  │  │    未来观察如果矛盾     → 权重衰减至遗忘                  │  │    │
│  │  └──────────────────────────────────────────────────────────┘  │    │
│  │                                                                 │    │
│  │  初始权重: w_init = σ(S_eff) mapped to [0.3, 0.5]              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   Maintenance Agent (低惊奇)                    │    │
│  │                                                                 │    │
│  │  触发条件: S_eff ≤ 1.0 (符合预期，强化信念)                    │    │
│  │                                                                 │    │
│  │  场景示例:                                                      │    │
│  │  已有知识: "用户擅长 Python"                                    │    │
│  │  新输入:   "我刚用 Python 写了一个 FastAPI 服务"                │    │
│  │                                                                 │    │
│  │  处理流程:                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────┐  │    │
│  │  │ 1. 识别匹配节点                                          │  │    │
│  │  │    检索到: "Python expert" (w=0.8)                       │  │    │
│  │  │                                                          │  │    │
│  │  │ 2. 强化权重                                              │  │    │
│  │  │    w_new = 0.8 + 0.05 × (1 - 0.8) = 0.81                │  │    │
│  │  │                                                          │  │    │
│  │  │ 3. 检查假设晋升                                          │  │    │
│  │  │    如果某 HYPOTHESIS 节点 w > 0.7:                       │  │    │
│  │  │    → 晋升为 FACT 节点                                    │  │    │
│  │  └──────────────────────────────────────────────────────────┘  │    │
│  │                                                                 │    │
│  │  权重更新公式: w_{t+1} = w_t + η × (1 - w_t)                   │    │
│  │  (渐近趋向 1.0，但永不超过)                                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 核心组件

| 组件 | 触发条件 | 操作 |
|-----|---------|------|
| `CorrectionAgent` | S_eff > θ_high | 诊断冲突、衰减旧节点、创建新事实 |
| `ProfilingAgent` | θ_low < S_eff ≤ θ_high | 生成假设节点、内部推理 |
| `MaintenanceAgent` | S_eff ≤ θ_low | 强化节点、假设晋升 |
| `SurpriseDispatcher` | - | 路由惊奇度包到对应代理 |

#### 关键代码

```python
# 惊奇度分发器
async def dispatch(self, packet, knowledge_graph):
    level = packet.get_surprise_level(self.theta_low, self.theta_high)

    if level == "high":
        result = await self.correction_agent.process(packet, knowledge_graph)
    elif level == "medium":
        result = await self.profiling_agent.process(packet, knowledge_graph)
    else:
        result = await self.maintenance_agent.process(packet, knowledge_graph)

    return {"level": level, "agent": agent_name, "result": result}
```

---

## 数学基础

### 惊奇度计算

```
原始惊奇度 (NLL):
S_raw(u_t) = -log P_LM(u_t | C_retrieved)

检索熵 (衡量检索质量):
H(C_t) = -Σ ŝ(ε) × log(ŝ(ε))

有效惊奇度 (区分冲突和无知):
S_eff(u_t) = S_raw(u_t) × (1 - λ × H(C_t))
```

### 贝叶斯权重更新

```
强化 (低惊奇):
w_{t+1} = w_t + η × (1 - w_t)

衰减 (高惊奇):
w_{t+1} = w_t × e^{-β × S_eff}

时间衰减 (假设节点):
w_{t+k} = w_t × e^{-γ × Δt}
```

### 意图掩码检索

```
Score(ε_k) = sim(emb(u_t), emb(ε_k)) × P(d(ε_k) | I_t) × w_k
              ↑ 语义相似度           ↑ 意图相关性      ↑ 置信权重
```

---

## 快速开始

### 1. 安装依赖

```bash
cd ProCoMemory
pip install -r requirements.txt
```

### 2. 配置 API

`config.yaml` 已预配置好 API，可直接使用。如需修改：

```yaml
model:
  openai_api_key: "your-api-key"
  openai_base_url: "https://your-api-endpoint/v1"
  llm_model: "gpt-4o"
```

### 3. 运行演示

```bash
# 交互式演示
python main.py

# 预定义场景演示
python main.py --scenario

# Mock 模式（不调用 API）
python main.py --mock
```

### 4. 运行 LoComo 评估

```bash
# 运行单个样本
python src/evaluation/run_experiment.py --sample 0

# 运行所有样本
python src/evaluation/run_experiment.py

# Mock 模式测试
python src/evaluation/run_experiment.py --mock --max-samples 2

# 基础分析                                                                                                               
  python analyze.py                                                                                                        
                                                                                                                           
  # 分析特定实验                                                                                                           
  python analyze.py --experiment my_experiment                                                                             
                                                                                                                           
  # 保存结果到 JSON                                                                                                        
  python analyze.py --output ./analysis_report.json                                                                        
                                                                                                                           
  # 导出为 CSV                                                                                                             
  python analyze.py --format csv --output ./results.csv                                                                    
                                                                                                                           
  # 对比多个实验                                                                                                           
  python analyze.py --compare exp1 exp2                                                                                    
                                                                                                                           
  # Surprisal 分布分析                                                                                                     
  python analyze.py --surprisal                                                                                            
                                                                                                                           
  # 详细错误分析                                                                                                           
  python analyze.py --errors --error-threshold 0.3
```

---

## 配置说明

| 配置项 | 默认值 | 说明 |
|-------|-------|------|
| `thresholds.theta_high` | 3.0 | 高惊奇阈值 → Correction Agent |
| `thresholds.theta_low` | 1.0 | 低惊奇阈值 → Maintenance Agent |
| `thresholds.lambda_factor` | 0.5 | 检索熵权重 |
| `weights.eta` | 0.05 | 强化学习率 |
| `weights.beta` | 0.3 | 衰减因子 |
| `working_memory.max_context_tokens` | 2000 | 工作记忆最大 token |

---

## 评估框架

### LoComo Benchmark

- **数据集**：10 个多轮对话样本，每个包含 50+ 轮对话和 30+ 个问答对
- **问题类别**：
  - Category 1: 单跳事实问题
  - Category 2: 时间推理问题
  - Category 3: 多跳推理问题
  - Category 4: 对抗性问题

### 评估指标

| 指标 | 说明 |
|-----|------|
| Exact Match (EM) | 标准化后精确匹配 |
| Token F1 | 词级别 F1 分数 |
| Contains Match | 包含匹配（更宽松） |
| Numeric Match | 数值匹配（适合时间问题） |

---

## 项目结构

```
ProCoMemory/
├── config.yaml              # 主配置文件
├── config.yaml.example      # 配置模板
├── config.py                # 配置加载模块
├── main.py                  # 主入口程序
├── requirements.txt         # 依赖列表
├── README.md                # 本文档
│
├── data/
│   ├── locomo10.json        # LoComo 数据集
│   ├── knowledge_graph.gml  # 持久化图数据
│   └── chroma_db/           # 向量数据库
│
└── src/
    ├── __init__.py
    │
    ├── core/
    │   ├── __init__.py
    │   ├── types.py         # Pydantic 数据结构
    │   └── orchestrator.py  # 核心编排器 (PCMSystem)
    │
    ├── layers/
    │   ├── __init__.py
    │   ├── layer1_perception.py   # L1: 感知层
    │   ├── layer2_world_model.py  # L2: 世界模型
    │   └── layer3_evolution.py    # L3: 演化引擎
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── llm_client.py    # OpenAI API 封装
    │   ├── math_utils.py    # 数学公式实现
    │   └── metrics.py       # 惊奇度计算
    │
    └── evaluation/
        ├── __init__.py
        ├── dataset.py       # LoComo 数据集加载器
        ├── metrics.py       # 评估指标
        └── run_experiment.py # 实验运行器
```

