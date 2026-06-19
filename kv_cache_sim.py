from __future__ import annotations

import argparse
import bisect
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}. Install dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from exc


BlockId = Tuple[int, int]


POLICIES = ["FIFO", "LRU", "LFU", "Heuristic", "Learned"]
COLORS = {
    "FIFO": "#8a8f98",
    "LRU": "#1f77b4",
    "LFU": "#2ca02c",
    "Heuristic": "#ff7f0e",
    "Learned": "#d62728",
}

@dataclass(frozen=True)
class LocalityProfile:
    name: str
    recent_weight: float
    prefix_weight: float
    random_weight: float


@dataclass(frozen=True)
class SimConfig:
    num_requests: int = 32
    seq_len: int = 2048
    block_size: int = 16
    cache_capacity: int = 256
    accesses_per_step: int = 10
    recent_window: int = 8
    prefix_blocks: int = 4
    hit_latency: float = 1.0
    swap_latency: float = 10.0
    evict_latency: float = 2.0
    prefetch_latency: float = 0.2
    lookahead: int = 256
    mistake_window: int = 128
    num_heads: int = 16
    low_head_threshold: float = 0.35
    prefetch_budget: int = 1
    prefetch_threshold: float = 0.55
    request_burst_min: int = 1
    request_burst_max: int = 4
    request_arrival_spread: int = 64
    seed: int = 42
    profile: LocalityProfile = LocalityProfile("medium", 0.60, 0.25, 0.15)

    @property
    def num_blocks(self) -> int:
        return max(1, math.ceil(self.seq_len / self.block_size))


@dataclass(frozen=True)
class Access:
    request_id: int
    block_index: int
    progress_block: int

    @property
    def block(self) -> BlockId:
        return (self.request_id, self.block_index)


@dataclass
class BlockMeta:
    insert_time: int
    last_access: int
    access_count: int


@dataclass
class Metrics:
    experiment: str
    x_value: str
    policy: str
    hit_rate: float
    avg_latency: float
    evictions: int
    swap_ins: int
    prefetches: int
    mistake_rate: float
    total_accesses: int


class LearnedReuseModel:
    """Lightweight reuse-distance predictor used by the Learned Cache policy."""

    def __init__(self, seed: int) -> None:
        self.constant_distance: float | None = None
        self.model = make_pipeline(
            StandardScaler(),
            DecisionTreeRegressor(
                max_depth=8,
                min_samples_leaf=12,
                random_state=seed,
            ),
        )

    def fit(self, x: List[List[float]], y: List[float]) -> None:
        if not x:
            self.constant_distance = 1.0
            return
        labels = np.asarray(y, dtype=float)
        if len(set(labels.tolist())) < 2:
            self.constant_distance = float(labels[0])
            return
        self.model.fit(np.asarray(x, dtype=float), labels)

    def predict_reuse_distance(self, x: List[List[float]]) -> np.ndarray:
        if self.constant_distance is not None:
            return np.full(len(x), self.constant_distance, dtype=float)
        if not hasattr(self.model[-1], "tree_"):
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict(np.asarray(x, dtype=float))


def split_counts(total: int, weights: Sequence[float]) -> List[int]:
    raw = [total * w for w in weights]
    counts = [int(math.floor(v)) for v in raw]
    while sum(counts) < total:
        idx = max(range(len(raw)), key=lambda i: raw[i] - counts[i])
        counts[idx] += 1
    return counts


def head_activity_score(block: BlockId, progress_block: int, config: SimConfig) -> float:
    """Synthetic average attention-head activity for a KV block.

    Lower values represent blocks whose heads are mostly inactive, matching the
    sparsity signal described in the assignment.
    """
    request_id, block_index = block
    mixed = (
        (request_id + 1) * 1103515245
        + (block_index + 17) * 12345
        + config.seed * 2654435761
    ) & 0xFFFFFFFF
    base = (mixed % 10000) / 10000.0
    if base < config.low_head_threshold:
        activity = 0.08 + 0.35 * base / max(config.low_head_threshold, 1e-6)
    else:
        activity = 0.45 + 0.55 * (base - config.low_head_threshold) / (
            1.0 - config.low_head_threshold
        )

    distance_to_tail = max(0, progress_block - block_index)
    if block_index < config.prefix_blocks:
        activity += 0.12
    if distance_to_tail <= config.recent_window:
        activity += 0.10
    activity = min(1.0, max(0.0, activity))
    active_heads = max(1, round(activity * config.num_heads))
    return active_heads / max(1, config.num_heads)


def sparse_head_flag(block: BlockId, progress_block: int, config: SimConfig) -> float:
    return 1.0 if head_activity_score(block, progress_block, config) < 0.45 else 0.0


def sample_from(pool: Sequence[int], count: int, rng: random.Random) -> List[int]:
    if count <= 0 or not pool:
        return []
    if count <= len(pool):
        return rng.sample(list(pool), count)
    return [rng.choice(pool) for _ in range(count)]


def sample_weighted_blocks(
    pool: Sequence[int],
    count: int,
    rng: random.Random,
    request_id: int,
    progress_block: int,
    config: SimConfig,
) -> List[int]:
    if count <= 0 or not pool:
        return []

    available = list(pool)
    chosen: List[int] = []
    for _ in range(min(count, len(available))):
        weights = [
            max(0.02, head_activity_score((request_id, block_index), progress_block, config))
            for block_index in available
        ]
        selected = rng.choices(available, weights=weights, k=1)[0]
        chosen.append(selected)
        available.remove(selected)

    while len(chosen) < count:
        weights = [
            max(0.02, head_activity_score((request_id, block_index), progress_block, config))
            for block_index in pool
        ]
        chosen.append(rng.choices(list(pool), weights=weights, k=1)[0])
    return chosen


def generate_trace(config: SimConfig) -> List[Access]:
    rng = random.Random(config.seed)
    trace: List[Access] = []
    counts = split_counts(
        config.accesses_per_step,
        [
            config.profile.recent_weight,
            config.profile.prefix_weight,
            config.profile.random_weight,
        ],
    )

    request_progress = [0 for _ in range(config.num_requests)]
    arrival_times = [
        rng.randint(0, max(0, config.request_arrival_spread))
        for _ in range(config.num_requests)
    ]
    if arrival_times:
        arrival_times[0] = 0

    schedule_tick = 0
    while any(progress < config.num_blocks for progress in request_progress):
        active_requests = [
            request_id
            for request_id, progress in enumerate(request_progress)
            if progress < config.num_blocks and arrival_times[request_id] <= schedule_tick
        ]
        if not active_requests:
            schedule_tick += 1
            continue

        request_id = rng.choice(active_requests)
        burst = rng.randint(config.request_burst_min, config.request_burst_max)
        for _ in range(burst):
            progress_block = request_progress[request_id]
            if progress_block >= config.num_blocks:
                break

            recent_start = max(0, progress_block - config.recent_window)
            recent = list(range(recent_start, progress_block + 1))
            prefix = list(range(0, min(config.prefix_blocks, progress_block + 1)))
            middle_start = min(config.prefix_blocks, progress_block + 1)
            middle_end = max(middle_start, recent_start)
            middle = list(range(middle_start, middle_end))

            chosen: List[int] = []
            chosen.extend(sample_weighted_blocks(recent, counts[0], rng, request_id, progress_block, config))
            chosen.extend(sample_weighted_blocks(prefix, counts[1], rng, request_id, progress_block, config))
            chosen.extend(sample_weighted_blocks(middle, counts[2], rng, request_id, progress_block, config))

            # Ensure the newly produced block enters the simulated KV cache.
            chosen.append(progress_block)

            for block_index in dict.fromkeys(chosen):
                trace.append(Access(request_id, block_index, progress_block))

            request_progress[request_id] += 1
            schedule_tick += 1
    return trace


def build_positions(trace: Sequence[Access]) -> Dict[BlockId, List[int]]:
    positions: Dict[BlockId, List[int]] = defaultdict(list)
    for idx, access in enumerate(trace):
        positions[access.block].append(idx)
    return positions


def next_access_after(
    positions: Dict[BlockId, List[int]], block: BlockId, current_time: int
) -> int | None:
    block_positions = positions.get(block)
    if not block_positions:
        return None
    idx = bisect.bisect_right(block_positions, current_time)
    if idx >= len(block_positions):
        return None
    return block_positions[idx]


def feature_vector(
    block: BlockId,
    current_time: int,
    progress_by_request: Dict[int, int],
    meta: Dict[BlockId, BlockMeta],
    config: SimConfig,
) -> List[float]:
    request_id, block_index = block
    block_meta = meta.get(block)
    if block_meta is None:
        last_gap = current_time + 1
        access_count = 0
        resident_age = current_time + 1
    else:
        last_gap = current_time - block_meta.last_access
        access_count = block_meta.access_count
        resident_age = current_time - block_meta.insert_time

    progress = progress_by_request.get(request_id, block_index)
    distance_to_tail = max(0, progress - block_index)
    active_context = max(1, progress + 1)
    position_ratio = block_index / active_context
    prefix = 1.0 if block_index < config.prefix_blocks else 0.0
    recent = 1.0 if distance_to_tail <= config.recent_window else 0.0
    head_activity = head_activity_score(block, progress, config)
    sparse_head = sparse_head_flag(block, progress, config)

    return [
        math.log1p(max(0, last_gap)),
        math.log1p(max(0, access_count)),
        math.log1p(max(0, resident_age)),
        distance_to_tail / max(1, config.num_blocks),
        position_ratio,
        prefix,
        recent,
        head_activity,
        sparse_head,
    ]


def build_training_data(
    trace: Sequence[Access],
    positions: Dict[BlockId, List[int]],
    config: SimConfig,
    max_examples: int = 24000,
) -> Tuple[List[List[float]], List[float]]:
    rng = random.Random(config.seed + 17)
    progress_by_request: Dict[int, int] = {}
    meta: Dict[BlockId, BlockMeta] = {}
    seen: List[BlockId] = []
    seen_set: set[BlockId] = set()
    stride = max(1, len(trace) // 1800)
    x_rows: List[List[float]] = []
    y_rows: List[float] = []

    for current_time, access in enumerate(trace):
        progress_by_request[access.request_id] = access.progress_block
        block = access.block
        if block not in seen_set:
            seen_set.add(block)
            seen.append(block)
            meta[block] = BlockMeta(current_time, current_time, 0)

        block_meta = meta[block]
        block_meta.last_access = current_time
        block_meta.access_count += 1

        if current_time % stride != 0:
            continue

        candidates = [block]
        if seen:
            candidates.extend(rng.sample(seen, min(8, len(seen))))

        for candidate in dict.fromkeys(candidates):
            next_idx = next_access_after(positions, candidate, current_time)
            if next_idx is None:
                label = float(config.lookahead * 2)
            else:
                label = float(min(next_idx - current_time, config.lookahead * 2))
            x_rows.append(feature_vector(candidate, current_time, progress_by_request, meta, config))
            y_rows.append(label)
            if len(x_rows) >= max_examples:
                return x_rows, y_rows

    return x_rows, y_rows


def train_learned_model(config: SimConfig) -> LearnedReuseModel:
    train_config = SimConfig(
        num_requests=config.num_requests,
        seq_len=config.seq_len,
        block_size=config.block_size,
        cache_capacity=config.cache_capacity,
        accesses_per_step=config.accesses_per_step,
        recent_window=config.recent_window,
        prefix_blocks=config.prefix_blocks,
        hit_latency=config.hit_latency,
        swap_latency=config.swap_latency,
        evict_latency=config.evict_latency,
        prefetch_latency=config.prefetch_latency,
        lookahead=config.lookahead,
        mistake_window=config.mistake_window,
        num_heads=config.num_heads,
        low_head_threshold=config.low_head_threshold,
        prefetch_budget=config.prefetch_budget,
        prefetch_threshold=config.prefetch_threshold,
        request_burst_min=config.request_burst_min,
        request_burst_max=config.request_burst_max,
        request_arrival_spread=config.request_arrival_spread,
        seed=config.seed + 1009,
        profile=config.profile,
    )
    train_trace = generate_trace(train_config)
    positions = build_positions(train_trace)
    x_train, y_train = build_training_data(train_trace, positions, train_config)
    model = LearnedReuseModel(seed=config.seed)
    model.fit(x_train, y_train)
    return model


def heuristic_score(
    block: BlockId,
    current_time: int,
    progress_by_request: Dict[int, int],
    meta: Dict[BlockId, BlockMeta],
    config: SimConfig,
) -> float:
    features = feature_vector(block, current_time, progress_by_request, meta, config)
    (
        last_gap,
        count,
        _resident_age,
        distance,
        _position_ratio,
        prefix,
        recent,
        head_activity,
        sparse_head,
    ) = features
    recency_score = 1.0 / (1.0 + last_gap)
    frequency_score = min(1.0, count / math.log1p(config.num_blocks * 4))
    distance_score = 1.0 / (1.0 + distance * config.num_blocks / max(1, config.recent_window))
    sparse_penalty = 0.08 * sparse_head
    score = (
        0.34 * recent
        + 0.22 * prefix
        + 0.18 * frequency_score
        + 0.14 * head_activity
        + 0.12 * (recency_score + distance_score) / 2.0
        - sparse_penalty
    )
    return max(0.0, score)


def learned_importance_scores(
    blocks: Sequence[BlockId],
    current_time: int,
    progress_by_request: Dict[int, int],
    meta: Dict[BlockId, BlockMeta],
    config: SimConfig,
    model: LearnedReuseModel,
) -> np.ndarray:
    features = [
        feature_vector(block, current_time, progress_by_request, meta, config)
        for block in blocks
    ]
    predicted_distance = model.predict_reuse_distance(features)
    distance_score = 1.0 / (
        1.0 + np.clip(predicted_distance, 0, config.lookahead * 2) / max(1, config.lookahead)
    )
    heuristic_scores = np.asarray(
        [
            heuristic_score(block, current_time, progress_by_request, meta, config)
            for block in blocks
        ],
        dtype=float,
    )
    head_scores = np.asarray(
        [
            head_activity_score(
                block,
                progress_by_request.get(block[0], block[1]),
                config,
            )
            for block in blocks
        ],
        dtype=float,
    )
    return 0.55 * distance_score + 0.30 * heuristic_scores + 0.15 * head_scores


def choose_victim(
    policy: str,
    cache: set[BlockId],
    current_time: int,
    progress_by_request: Dict[int, int],
    meta: Dict[BlockId, BlockMeta],
    config: SimConfig,
    model: LearnedReuseModel | None,
) -> BlockId:
    if policy == "FIFO":
        return min(cache, key=lambda b: (meta[b].insert_time, meta[b].last_access, b))
    if policy == "LRU":
        return min(cache, key=lambda b: (meta[b].last_access, meta[b].insert_time, b))
    if policy == "LFU":
        return min(cache, key=lambda b: (meta[b].access_count, meta[b].last_access, b))
    if policy == "Heuristic":
        return min(
            cache,
            key=lambda b: (
                heuristic_score(b, current_time, progress_by_request, meta, config),
                meta[b].last_access,
                b,
            ),
        )
    if policy == "Learned":
        if model is None:
            raise ValueError("Learned policy requires a trained model.")
        blocks = list(cache)
        importance = learned_importance_scores(
            blocks, current_time, progress_by_request, meta, config, model
        )
        min_idx = int(np.argmin(importance))
        return blocks[min_idx]
    raise ValueError(f"Unknown policy: {policy}")


def prefetch_for_learned(
    access: Access,
    cache: set[BlockId],
    current_time: int,
    progress_by_request: Dict[int, int],
    meta: Dict[BlockId, BlockMeta],
    positions: Dict[BlockId, List[int]],
    config: SimConfig,
    model: LearnedReuseModel | None,
) -> Tuple[int, int, int, float]:
    if model is None or config.prefetch_budget <= 0:
        return 0, 0, 0, 0.0

    request_id = access.request_id
    progress = access.progress_block
    prefix_pool = list(range(0, min(config.prefix_blocks, progress + 1)))
    recent_pool = list(range(max(0, progress - config.recent_window), progress + 1))
    middle_start = min(config.prefix_blocks, progress + 1)
    middle_end = max(middle_start, progress - config.recent_window)
    middle_pool = list(range(middle_start, middle_end, max(1, config.recent_window // 2)))
    candidate_blocks = [
        (request_id, block_index)
        for block_index in dict.fromkeys(prefix_pool + recent_pool + middle_pool)
        if (request_id, block_index) not in cache
    ]
    if not candidate_blocks:
        return 0, 0, 0, 0.0

    scores = learned_importance_scores(
        candidate_blocks, current_time, progress_by_request, meta, config, model
    )
    ranked = sorted(
        zip(candidate_blocks, scores.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    prefetches = 0
    evictions = 0
    bad_evictions = 0
    latency = 0.0
    for candidate, candidate_score in ranked[: config.prefetch_budget]:
        if candidate_score < config.prefetch_threshold:
            continue

        if len(cache) >= config.cache_capacity:
            victim = choose_victim(
                "Learned", cache, current_time, progress_by_request, meta, config, model
            )
            victim_score = learned_importance_scores(
                [victim], current_time, progress_by_request, meta, config, model
            )[0]
            if candidate_score <= victim_score:
                continue
            cache.remove(victim)
            evictions += 1
            next_idx = next_access_after(positions, victim, current_time)
            if next_idx is not None and next_idx <= current_time + config.mistake_window:
                bad_evictions += 1

        cache.add(candidate)
        if candidate not in meta:
            meta[candidate] = BlockMeta(current_time, current_time, 0)
        else:
            meta[candidate].insert_time = current_time
        prefetches += 1
        latency += config.prefetch_latency

    return prefetches, evictions, bad_evictions, latency


def simulate(
    trace: Sequence[Access],
    config: SimConfig,
    policy: str,
    model: LearnedReuseModel | None = None,
) -> Metrics:
    cache: set[BlockId] = set()
    meta: Dict[BlockId, BlockMeta] = {}
    progress_by_request: Dict[int, int] = {}
    positions = build_positions(trace)
    hits = 0
    misses = 0
    evictions = 0
    swap_ins = 0
    prefetches = 0
    bad_evictions = 0
    total_latency = 0.0

    for current_time, access in enumerate(trace):
        progress_by_request[access.request_id] = access.progress_block
        block = access.block

        if block in cache:
            hits += 1
            total_latency += config.hit_latency
        else:
            misses += 1
            swap_ins += 1
            total_latency += config.swap_latency
            if len(cache) >= config.cache_capacity:
                victim = choose_victim(
                    policy, cache, current_time, progress_by_request, meta, config, model
                )
                cache.remove(victim)
                evictions += 1
                total_latency += config.evict_latency
                next_idx = next_access_after(positions, victim, current_time)
                if next_idx is not None and next_idx <= current_time + config.mistake_window:
                    bad_evictions += 1
            cache.add(block)
            if block not in meta:
                meta[block] = BlockMeta(current_time, current_time, 0)
            meta[block].insert_time = current_time

        if block not in meta:
            meta[block] = BlockMeta(current_time, current_time, 0)
        meta[block].last_access = current_time
        meta[block].access_count += 1

        if policy == "Learned":
            (
                new_prefetches,
                prefetch_evictions,
                prefetch_bad_evictions,
                prefetch_latency,
            ) = prefetch_for_learned(
                access,
                cache,
                current_time,
                progress_by_request,
                meta,
                positions,
                config,
                model,
            )
            prefetches += new_prefetches
            evictions += prefetch_evictions
            bad_evictions += prefetch_bad_evictions
            total_latency += prefetch_latency

    total = hits + misses
    return Metrics(
        experiment="",
        x_value="",
        policy=policy,
        hit_rate=hits / total if total else 0.0,
        avg_latency=total_latency / total if total else 0.0,
        evictions=evictions,
        swap_ins=swap_ins,
        prefetches=prefetches,
        mistake_rate=bad_evictions / evictions if evictions else 0.0,
        total_accesses=total,
    )


def with_config(base: SimConfig, **updates: object) -> SimConfig:
    data = base.__dict__.copy()
    data.update(updates)
    return SimConfig(**data)


def evaluate_setting(
    experiment: str,
    x_value: str,
    config: SimConfig,
    train_cache_capacity: int | None = None,
    model: LearnedReuseModel | None = None,
) -> List[Metrics]:
    print(f"Running {experiment}={x_value} ...", flush=True)
    train_config = config
    if train_cache_capacity is not None:
        train_config = with_config(config, cache_capacity=train_cache_capacity)
    if model is None:
        model = train_learned_model(train_config)
    trace = generate_trace(config)
    rows: List[Metrics] = []
    for policy in POLICIES:
        result = simulate(trace, config, policy, model if policy == "Learned" else None)
        result.experiment = experiment
        result.x_value = x_value
        rows.append(result)
    return rows


def run_experiments(base: SimConfig, quick: bool = False, extended: bool = False) -> List[Metrics]:
    if extended:
        capacities = [64, 128, 256, 512]
        lengths = [512, 1024, 2048, 4096]
    elif quick:
        capacities = [128, 256]
        lengths = [1024, 2048]
    else:
        capacities = [64, 128, 256]
        lengths = [512, 1024, 2048]

    profiles = [
        LocalityProfile("strong", 0.80, 0.15, 0.05),
        LocalityProfile("medium", 0.60, 0.25, 0.15),
        LocalityProfile("weak", 0.40, 0.30, 0.30),
    ]
    if quick:
        profiles = profiles[:2]

    all_rows: List[Metrics] = []
    model_cache: Dict[Tuple[object, ...], LearnedReuseModel] = {}

    def cached_model(config: SimConfig) -> LearnedReuseModel:
        key = (
            config.num_requests,
            config.seq_len,
            config.block_size,
            config.accesses_per_step,
            config.recent_window,
            config.prefix_blocks,
            config.lookahead,
            config.num_heads,
            config.low_head_threshold,
            config.request_burst_min,
            config.request_burst_max,
            config.request_arrival_spread,
            config.seed,
            config.profile.name,
            config.profile.recent_weight,
            config.profile.prefix_weight,
            config.profile.random_weight,
        )
        if key not in model_cache:
            model_cache[key] = train_learned_model(config)
        return model_cache[key]

    for capacity in capacities:
        config = with_config(base, cache_capacity=capacity, seq_len=2048, profile=profiles[1])
        all_rows.extend(
            evaluate_setting(
                "capacity",
                str(capacity),
                config,
                train_cache_capacity=capacity,
                model=cached_model(config),
            )
        )

    for seq_len in lengths:
        capacity = min(base.cache_capacity, max(64, math.ceil(seq_len / base.block_size) * 3))
        config = with_config(base, seq_len=seq_len, cache_capacity=capacity, profile=profiles[1])
        all_rows.extend(evaluate_setting("seq_len", str(seq_len), config, model=cached_model(config)))

    for profile in profiles:
        config = with_config(base, seq_len=2048, cache_capacity=base.cache_capacity, profile=profile)
        all_rows.extend(
            evaluate_setting("locality", profile.name, config, model=cached_model(config))
        )

    return all_rows


def metrics_dataframe(rows: Sequence[Metrics]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment": row.experiment,
                "x_value": row.x_value,
                "policy": row.policy,
                "hit_rate": row.hit_rate,
                "avg_latency": row.avg_latency,
                "evictions": row.evictions,
                "swap_ins": row.swap_ins,
                "prefetches": row.prefetches,
                "mistake_rate": row.mistake_rate,
                "total_accesses": row.total_accesses,
            }
            for row in rows
        ]
    )


def write_csv(rows: Sequence[Metrics], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dataframe(rows).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8")


def write_metric_plot(
    df: pd.DataFrame,
    experiment: str,
    metric: str,
    title: str,
    output_stem: str,
    y_label: str,
    output_dir: Path,
) -> None:
    selected = df[df["experiment"] == experiment].copy()
    if selected.empty:
        return

    x_values = list(dict.fromkeys(selected["x_value"].tolist()))
    x_pos = np.arange(len(x_values))
    plt.figure(figsize=(9.2, 5.4), dpi=140)

    for policy in POLICIES:
        policy_rows = selected[selected["policy"] == policy]
        values = []
        for x_value in x_values:
            matched = policy_rows[policy_rows["x_value"] == x_value]
            values.append(float(matched[metric].iloc[0]) if not matched.empty else np.nan)
        plt.plot(
            x_pos,
            values,
            marker="o",
            linewidth=2.2,
            markersize=5,
            label=policy,
            color=COLORS.get(policy),
        )

    plt.title(title)
    plt.xlabel(experiment)
    plt.ylabel(y_label)
    plt.xticks(x_pos, x_values)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{output_stem}.svg")
    plt.savefig(output_dir / f"{output_stem}.png")
    plt.close()


def write_plots(rows: Sequence[Metrics], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = metrics_dataframe(rows)
    write_metric_plot(
        df,
        "capacity",
        "hit_rate",
        "Cache Capacity vs Hit Rate",
        "capacity_hit_rate",
        "Hit rate",
        output_dir,
    )
    write_metric_plot(
        df,
        "capacity",
        "avg_latency",
        "Cache Capacity vs Average Latency",
        "capacity_avg_latency",
        "Average latency",
        output_dir,
    )
    write_metric_plot(
        df,
        "seq_len",
        "avg_latency",
        "Sequence Length vs Average Latency",
        "seq_len_avg_latency",
        "Average latency",
        output_dir,
    )
    write_metric_plot(
        df,
        "locality",
        "mistake_rate",
        "Locality Profile vs Mistake Eviction Rate",
        "locality_mistake_rate",
        "Mistake rate",
        output_dir,
    )


def print_main_table(rows: Sequence[Metrics]) -> None:
    selected = [r for r in rows if r.experiment == "capacity" and r.x_value == "256"]
    if not selected:
        selected = list(rows[: len(POLICIES)])
    print("\nMain comparison at cache capacity = 256 blocks")
    print("Policy      HitRate   AvgLatency   Evictions   SwapIns   Prefetches   MistakeRate")
    for row in selected:
        print(
            f"{row.policy:<10}  {row.hit_rate:>7.3f}   {row.avg_latency:>10.3f}"
            f"   {row.evictions:>9}   {row.swap_ins:>7}   {row.prefetches:>10}"
            f"   {row.mistake_rate:>11.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KV Cache replacement simulator for LLM inference."
    )
    parser.add_argument("--output-dir", default="results", help="Directory for CSV and SVG files.")
    parser.add_argument("--num-requests", type=int, default=32, help="Number of concurrent requests.")
    parser.add_argument("--cache-capacity", type=int, default=256, help="Default GPU cache blocks.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--quick", action="store_true", help="Run fewer experiment points.")
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Include heavier 4096-token and 512-block settings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    base = SimConfig(
        num_requests=args.num_requests,
        cache_capacity=args.cache_capacity,
        seed=args.seed,
    )
    rows = run_experiments(base, quick=(args.quick or not args.extended), extended=args.extended)
    write_csv(rows, output_dir)
    write_plots(rows, output_dir)
    print_main_table(rows)
    print(f"\nWrote results to: {output_dir.resolve()}")
    print("CSV: summary.csv")
    print("SVG: capacity_hit_rate.svg, capacity_avg_latency.svg, seq_len_avg_latency.svg, locality_mistake_rate.svg")


if __name__ == "__main__":
    main()
