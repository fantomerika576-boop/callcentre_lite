"""Завантаження журналу дзвінків та аналіз вхідних даних."""
from __future__ import annotations

import os
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

OPEN_HOUR, HOURS = 8, 10                 # робочий день 08:00–18:00
DAY = HOURS * 3600.0                     # тривалість дня, с
MONTHS = ["Січень", "Лютий", "Березень", "Квітень", "Травень", "Червень",
          "Липень", "Серпень", "Вересень", "Жовтень", "Листопад", "Грудень"]
COLUMNS = ["call_id", "date", "daily_caller", "call_started", "call_answered",
           "call_ended", "wait_length", "service_length", "meets_standard"]


def _seconds(col: pd.Series) -> np.ndarray:
    """'8:02:42 AM' або '08:02:42' -> секунди від опівночі."""
    for fmt in ("%I:%M:%S %p", "%H:%M:%S"):
        try:
            t = pd.to_datetime(col, format=fmt)
        except (ValueError, TypeError):
            continue
        return (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).to_numpy(float)
    raise ValueError("Не вдалося розпізнати формат часу у журналі.")


def _agents(df: pd.DataFrame) -> int:
    """Кількість операторів = максимум одночасних розмов (метод розгортки)."""
    day = (df["date"] - df["date"].min()).dt.days.to_numpy(float) * 1e5   # окремий «відрізок» для кожного дня
    t = np.concatenate([day + df["start"].to_numpy(), day + df["end"].to_numpy()])
    d = np.concatenate([np.ones(len(df)), -np.ones(len(df))])              # +1 початок розмови, −1 кінець
    order = np.lexsort((d, t))                                             # при рівному часі спершу «−1»
    return int(np.cumsum(d[order]).max())


class Log:
    """Журнал дзвінків + зведені показники."""

    def __init__(self, raw: pd.DataFrame, source: str = ""):
        miss = [c for c in COLUMNS if c not in raw.columns]
        if miss:
            raise ValueError("У файлі немає стовпців: " + ", ".join(miss))
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"])
        t0 = OPEN_HOUR * 3600
        df["arr"] = _seconds(df["call_started"]) - t0       # надходження, с від 08:00
        df["start"] = _seconds(df["call_answered"]) - t0    # початок розмови
        df["end"] = _seconds(df["call_ended"]) - t0         # завершення
        df["month"] = df["date"].dt.month
        df["hour"] = np.clip((df["arr"] // 3600).astype(int), 0, HOURS - 1)
        df["ok"] = df["meets_standard"].astype(str).str.strip().str.upper().isin(["TRUE", "1"])
        self.df = df.sort_values(["date", "arr", "call_id"]).reset_index(drop=True)
        self.source = source
        self.days = int(df["date"].nunique())
        self._days = df.groupby("month")["date"].nunique().to_dict()
        self.agents = _agents(self.df)
        self.service_mean = float(df["service_length"].mean())
        self.sla = float(df.loc[df["ok"], "wait_length"].max())            # поріг SLA з даних
        first = df.groupby("date")["arr"].min()
        self.first_at_open = bool((first == 0).mean() > 0.95)             # перший дзвінок дня рівно о 08:00

    def sub(self, month: int | None = None) -> pd.DataFrame:
        return self.df if month is None else self.df[self.df["month"] == month]

    def days_in(self, month: int | None = None) -> int:
        return self.days if month is None else int(self._days.get(month, 0))

    def months(self) -> list[int]:
        return sorted(self._days)

    def kpis(self, month: int | None = None) -> dict:
        d, nd = self.sub(month), max(self.days_in(month), 1)
        w = d["wait_length"]
        return dict(calls=len(d), days=nd, calls_day=len(d) / nd, wait_mean=float(w.mean()),
                    waited_pct=100 * float((w > 0).mean()), sla_pct=100 * float(d["ok"].mean()),
                    service_mean=float(d["service_length"].mean()),
                    util_pct=100 * float(d["service_length"].sum()) / (self.agents * DAY * nd))

    def hourly_per_day(self) -> np.ndarray:
        return np.bincount(self.df["hour"], minlength=HOURS)[:HOURS] / self.days

    def rates(self, month: int | None = None) -> np.ndarray:
        """Інтенсивність потоку λ(h), викл./год, з поправками на два артефакти журналу:
        1) перший дзвінок кожного дня фіксується рівно о 08:00:00 — він не з пуассонівського потоку;
        2) дзвінки, не завершені до 18:00, у журнал не потрапили — коригуємо останню годину."""
        d, nd = self.sub(month), max(self.days_in(month), 1)
        cnt = np.bincount(d["hour"], minlength=HOURS)[:HOURS].astype(float)
        if self.first_at_open:
            cnt[0] -= nd
        m = max(self.service_mean, 1.0)
        cnt[-1] /= 1.0 - (m / 3600.0) * (1.0 - np.exp(-3600.0 / m))     # ймовірність встигнути завершитись
        return cnt / nd

    def monthly(self) -> pd.DataFrame:
        rows = []
        for m in self.months():
            k = self.kpis(m)
            rows.append(dict(month=m, calls_day=k["calls_day"], wait_mean=k["wait_mean"],
                             sla_pct=k["sla_pct"], util_pct=k["util_pct"]))
        return pd.DataFrame(rows)


def load_csv(path: str) -> Log:
    """Читає CSV або ZIP із одним CSV-файлом."""
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
            if not names:
                raise ValueError("В архіві немає CSV-файлу.")
            with z.open(names[0]) as f:
                return Log(pd.read_csv(f), names[0])
    return Log(pd.read_csv(path), os.path.basename(path))


def analyze(log: Log) -> dict:
    """Ідентифікація моделі: закон обслуговування, характер потоку, тренд навантаження."""
    s = log.df["service_length"].to_numpy(float)
    s = s[s > 0]
    ia = log.df.groupby("date")["arr"].diff().dropna().to_numpy()
    mon = log.monthly()
    lr = stats.linregress(mon["month"], mon["calls_day"])
    return dict(service_cv=float(s.std() / s.mean()),
                gamma_shape=float(stats.gamma.fit(s, floc=0)[0]),
                ks_exp=float(stats.kstest(s, "expon", args=(0, s.mean())).statistic),
                ia_cv=float(ia.std() / ia.mean()),
                slope=float(lr.slope), intercept=float(lr.intercept), r2=float(lr.rvalue ** 2))


def findings(log: Log, an: dict) -> list[tuple[str, str]]:
    """Текстові висновки для вкладки «Дані»."""
    k = log.kpis()
    return [
        ("Набір даних", f"{k['calls']:,} викликів за {k['days']} робочих днів "
                        f"({log.df['date'].min():%d.%m.%Y} – {log.df['date'].max():%d.%m.%Y}).".replace(",", " ")),
        ("Кількість операторів", f"Максимум одночасних розмов = {log.agents}, тому в моделі N = {log.agents} "
                                 f"операторів і одна спільна черга (FIFO)."),
        ("Потік викликів", f"Коефіцієнт варіації інтервалів між дзвінками {an['ia_cv']:.2f} (для пуассонівського потоку ≈ 1). "
                           f"Потік моделюємо як пуассонівський з інтенсивністю λ(t), що залежить від години."),
        ("Обслуговування", f"Середня розмова {k['service_mean']:.0f} с, CV = {an['service_cv']:.2f}; форма гамма-розподілу "
                           f"≈ {an['gamma_shape']:.2f} (=1 для експоненційного), D Колмогорова = {an['ks_exp']:.3f}. "
                           f"Тому тривалість розмови моделюємо експоненційним законом."),
        ("Навантаження", f"Інтенсивність зростає на {an['slope']:+.1f} викликів/день щомісяця (R² = {an['r2']:.2f}) — "
                         f"це головна причина падіння якості обслуговування."),
        ("Якість (SLA)", f"Очікування ≤ {log.sla:.0f} с виконано для {k['sla_pct']:.1f}% викликів; "
                         f"чекали в черзі {k['waited_pct']:.1f}%, середнє очікування {k['wait_mean']:.1f} с."),
        ("Особливості журналу", "• перший дзвінок дня рівно о 08:00:00;\n"
                                "• дзвінки, не завершені до 18:00, відсутні (модель це враховує);\n"
                                "• відмов немає — терпіння клієнтів нескінченне."),
    ]
