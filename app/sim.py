"""Імітаційна модель контакт-центру (одна черга FIFO, N операторів) і формули Erlang C.

Бізнес-процес:  надходження виклику → черга → оператор (розмова) → перевірка SLA.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

HOURS = 10
DAY = HOURS * 3600.0                     # робочий день 08:00–18:00, с
GRID = np.arange(0.0, DAY, 300.0)        # моменти (кожні 5 хв), у які фіксуємо довжину черги


class Cancelled(Exception):
    """Користувач перервав обчислення."""


@dataclass
class Config:
    rates: tuple                         # λ(h): викликів/год для кожної години 08–18
    agents: int = 4                      # кількість операторів
    service_mean: float = 300.0          # середня тривалість розмови, с
    sla: float = 60.0                    # поріг SLA, с
    load: float = 1.0                    # множник навантаження (1 = як у даних)
    first_at_open: bool = True           # перший дзвінок дня рівно о 08:00:00 (як у журналі)
    days: int = 22                       # робочих днів в одному прогоні
    reps: int = 20                       # незалежних прогонів
    seed: int = 42


def config_from_log(log, month: int | None = None, **kw) -> Config:
    """Конфігурація, відкалібрована за журналом (потік, N, тривалість розмови, поріг SLA)."""
    return Config(rates=tuple(log.rates(month)), agents=log.agents, service_mean=log.service_mean,
                  sla=log.sla, first_at_open=log.first_at_open, **kw)


# ---------------------------------------------------------------- один робочий день
def simulate_day(cfg: Config, rng: np.random.Generator) -> dict:
    # 1. Потік надходжень: у кожній годині Poisson(λ) викликів, моменти рівномірні всередині години
    counts = rng.poisson(np.asarray(cfg.rates, float) * cfg.load)
    arr = np.sort(np.concatenate([h * 3600.0 + rng.random(k) * 3600.0 for h, k in enumerate(counts)]))
    if cfg.first_at_open:
        arr = np.concatenate(([0.0], arr))
    n = arr.size
    svc = rng.exponential(cfg.service_mean, n)                 # 2. тривалість розмов
    # 3. Обробка подій надходження. Календар — купа моментів звільнення операторів:
    #    виклик займає оператора, що звільнився найраніше; якщо всі зайняті — чекає (FIFO).
    free = [0.0] * cfg.agents
    heapq.heapify(free)
    start = np.empty(n)
    for i, a in enumerate(arr.tolist()):
        s = max(a, free[0])                                    # початок розмови
        start[i] = s
        heapq.heapreplace(free, s + svc[i])                    # оператор знову вільний після розмови
    counted = start + svc <= DAY                               # як у журналі: виклики, не завершені до 18:00, не враховуються
    wait = np.floor(start) - np.floor(arr)                     # очікування з точністю до секунди, як у журналі
    queue = np.searchsorted(arr, GRID, "right") - np.searchsorted(start, GRID, "right")   # довжина черги
    return dict(arr=arr, svc=svc, wait=wait, counted=counted, queue=queue)


# ---------------------------------------------------------------- серія прогонів
@dataclass
class Result:
    cfg: Config
    summary: dict                        # KPI -> {mean, sd, lo, hi}
    hourly: pd.DataFrame                 # показники за годинами
    waits: np.ndarray                    # усі очікування (для гістограми)
    queue: np.ndarray                    # (днів × моментів) довжина черги
    erlang: dict                         # аналітична оцінка для порівняння
    elapsed: float


def _ci(x) -> dict:
    x = np.asarray(x, float)
    m = float(x.mean())
    if len(x) < 2:
        return dict(mean=m, sd=0.0, lo=m, hi=m)
    sd = float(x.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(x) - 1) * sd / math.sqrt(len(x)))     # 95% довірчий інтервал
    return dict(mean=m, sd=sd, lo=m - half, hi=m + half)


def run(cfg: Config, progress=None, should_stop=None) -> Result:
    t0 = time.perf_counter()
    rows, waits, queues = [], [], []
    hour = np.zeros((3, HOURS))                                # викликів, сума очікувань, у межах SLA
    for r, ss in enumerate(np.random.SeedSequence(cfg.seed).spawn(cfg.reps)):
        if should_stop and should_stop():
            raise Cancelled()
        rng = np.random.default_rng(ss)
        w_rep, busy, qmax = [], 0.0, []
        for _ in range(cfg.days):
            d = simulate_day(cfg, rng)
            c = d["counted"]
            w = d["wait"][c]
            h = np.minimum((d["arr"][c] // 3600).astype(int), HOURS - 1)
            hour[0] += np.bincount(h, minlength=HOURS)
            hour[1] += np.bincount(h, weights=w, minlength=HOURS)
            hour[2] += np.bincount(h[w <= cfg.sla], minlength=HOURS)
            w_rep.append(w)
            busy += d["svc"][c].sum()
            qmax.append(d["queue"].max())
            queues.append(d["queue"])
        w = np.concatenate(w_rep)
        waits.append(w)
        has = w.size > 0
        rows.append(dict(calls_day=w.size / cfg.days,
                         wait_mean=float(w.mean()) if has else 0.0,
                         waited_pct=100 * float((w > 0).mean()) if has else 0.0,
                         sla_pct=100 * float((w <= cfg.sla).mean()) if has else 100.0,
                         p95=float(np.percentile(w, 95)) if has else 0.0,
                         util_pct=100 * busy / (cfg.agents * DAY * cfg.days),
                         q_max=float(np.mean(qmax))))
        if progress:
            progress((r + 1) / cfg.reps, f"Прогін {r + 1} із {cfg.reps}")
    rep = pd.DataFrame(rows)
    n = np.maximum(hour[0], 1)
    hourly = pd.DataFrame(dict(hour=np.arange(HOURS), calls_day=hour[0] / (cfg.days * cfg.reps),
                               wait_mean=hour[1] / n, sla_pct=100 * hour[2] / n))
    lam = np.asarray(cfg.rates, float) * cfg.load
    return Result(cfg=cfg, summary={k: _ci(rep[k]) for k in rep.columns}, hourly=hourly,
                  waits=np.concatenate(waits), queue=np.vstack(queues),
                  erlang=profile(lam, cfg.service_mean, cfg.agents, cfg.sla),
                  elapsed=time.perf_counter() - t0)


# ---------------------------------------------------------------- аналітична модель M/M/c (Erlang C)
def erlang_c(c: int, a: float) -> float:
    """Ймовірність очікування; c — операторів, a = λ·E[S] — навантаження в Ерлангах."""
    if a <= 0:
        return 0.0
    if a >= c:
        return 1.0
    b = 1.0                                                    # рекурсія Erlang B (стійка до великих c)
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b / (1.0 - (a / c) * (1.0 - b))


def mmc(lam_h: float, mean: float, c: int, sla: float) -> dict:
    lam = lam_h / 3600.0
    a = lam * mean
    if lam <= 0:
        return dict(a=0.0, p_wait=0.0, wq=0.0, sla=1.0, stable=True)
    if c < 1 or a >= c:
        return dict(a=a, p_wait=1.0, wq=math.inf, sla=0.0, stable=False)
    p = erlang_c(c, a)
    d = c / mean - lam                                         # c·μ − λ
    return dict(a=a, p_wait=p, wq=p / d, sla=1.0 - p * math.exp(-d * sla), stable=True)


def profile(rates, mean: float, c: int, sla: float) -> dict:
    """Кожна година — окрема стаціонарна система; показники усереднюються з вагою λ."""
    rows = [mmc(float(r), mean, c, sla) for r in rates]
    w = np.asarray(rates, float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(rows), 1 / len(rows))
    return dict(rho=sum(r["a"] for r in rows) / (c * len(rows)),
                p_wait=float(sum(wi * r["p_wait"] for wi, r in zip(w, rows))),
                wq=float(sum(wi * r["wq"] for wi, r in zip(w, rows))),
                sla=float(sum(wi * r["sla"] for wi, r in zip(w, rows))),
                stable=all(r["stable"] for r in rows))


def min_agents(lam_h: float, mean: float, target: float, sla: float) -> int:
    """Мінімальна кількість операторів, за якої частка викликів з очікуванням ≤ sla не менша за target."""
    c = max(1, int(lam_h / 3600.0 * mean) + 1)
    while c < 80 and mmc(lam_h, mean, c, sla)["sla"] < target:
        c += 1
    return c


# ---------------------------------------------------------------- експерименти
def sweep(cfg: Config, values, progress=None, should_stop=None) -> pd.DataFrame:
    """Підбір штату: серія симуляцій для різної кількості операторів."""
    rows = []
    for i, n in enumerate(values):
        if should_stop and should_stop():
            raise Cancelled()
        r = run(replace(cfg, agents=n))
        s = r.summary
        rows.append(dict(agents=n, util=s["util_pct"]["mean"], wait=s["wait_mean"]["mean"], p95=s["p95"]["mean"],
                         sla=s["sla_pct"]["mean"], sla_lo=s["sla_pct"]["lo"], sla_hi=s["sla_pct"]["hi"],
                         sla_erl=100 * r.erlang["sla"]))
        if progress:
            progress((i + 1) / len(values), f"Операторів: {n}")
    return pd.DataFrame(rows)


def validate(log, cfg: Config, progress=None, should_stop=None) -> pd.DataFrame:
    """Валідація: для кожного місяця порівнюємо факт із результатом імітації."""
    rows, months = [], log.months()
    for i, m in enumerate(months):
        if should_stop and should_stop():
            raise Cancelled()
        c = replace(cfg, rates=tuple(log.rates(m)), days=log.days_in(m), load=1.0, seed=cfg.seed + m)
        s, k = run(c).summary, log.kpis(m)
        rows.append(dict(month=m, calls_act=k["calls_day"], calls_sim=s["calls_day"]["mean"],
                         wait_act=k["wait_mean"], wait_sim=s["wait_mean"]["mean"], wait_sd=s["wait_mean"]["sd"],
                         sla_act=k["sla_pct"], sla_sim=s["sla_pct"]["mean"], sla_sd=s["sla_pct"]["sd"],
                         util_act=k["util_pct"], util_sim=s["util_pct"]["mean"]))
        if progress:
            progress((i + 1) / len(months), f"Валідація: місяць {m}")
    return pd.DataFrame(rows)


def mape(act, sim) -> float:
    """Середня абсолютна відсоткова похибка, %."""
    act, sim = np.asarray(act, float), np.asarray(sim, float)
    return float(np.mean(np.abs(sim - act) / np.maximum(np.abs(act), 1e-9)) * 100)
