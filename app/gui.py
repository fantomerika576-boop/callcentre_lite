"""Графічний інтерфейс: вкладки «Дані», «Симуляція», «Підбір штату», «Валідація»."""
from __future__ import annotations

import math
import os
import queue
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, font as tkfont, messagebox, ttk

import warnings

import matplotlib

warnings.filterwarnings("ignore", message=".*layoutgrids.*")
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import sim  # noqa: E402
from .data import HOURS, MONTHS, analyze, findings, load_csv  # noqa: E402

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                            "axes.spines.top": False, "axes.spines.right": False,
                            "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
                            "axes.facecolor": "white", "axes.edgecolor": "#CBD5E1",
                            "axes.labelcolor": "#334155", "text.color": "#334155",
                            "xtick.color": "#64748B", "ytick.color": "#64748B"})
BLUE, ORANGE, GREEN, RED, GRAY = "#2563EB", "#D97706", "#059669", "#DC2626", "#6B7280"

# ---- палітра інтерфейсу (білий фон, темний текст, синій акцент) ----
BG = "#FFFFFF"          # фон вікна та карток
PANEL = "#F8FAFC"       # легкий фон смуг/панелей
BORDER = "#E2E8F0"      # тонкі межі
TEXT = "#1E293B"        # основний текст
MUTED2 = "#64748B"      # другорядний текст
ACCENT = "#2563EB"      # акцентний синій (головні кнопки)
ACCENT_DARK = "#1D4ED8"
ACCENT_LIGHT = "#EFF6FF"
DANGER = "#DC2626"
DANGER_DARK = "#B91C1C"
DEFAULT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "simulated_call_centre.csv")
KPI_NAMES = {"calls_day": "Викликів за день", "wait_mean": "Середнє очікування, с", "waited_pct": "Чекали в черзі, %",
             "sla_pct": "SLA виконано, %", "p95": "P95 очікування, с", "util_pct": "Завантаження операторів, %",
             "q_max": "Макс. довжина черги"}


# ================================================================= допоміжні віджети
class _Toolbar(NavigationToolbar2Tk):
    """Панель інструментів без напису з координатами курсора; білий фон під тему."""

    def set_message(self, s):
        pass                                                    # координати при наведенні на графік не показуємо

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent, pack_toolbar=False)
        self.configure(background=BG, highlightthickness=0, bd=0)
        for child in self.winfo_children():
            try:
                child.configure(background=BG)
            except tk.TclError:
                pass


class Chart(ttk.Frame):
    """Область з графіком matplotlib і панеллю інструментів (збереження в PNG)."""

    def __init__(self, parent, size=(8, 4.5), hint="Тут з'являться графіки"):
        super().__init__(parent)
        self.fig = Figure(figsize=size, dpi=100, layout="constrained")   # поля підбираються автоматично при кожному перемальовуванні
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        _Toolbar(self.canvas, self).pack(side="bottom", fill="x")
        widget = self.canvas.get_tk_widget()
        widget.configure(width=200, height=160)                 # мінімальний запит; реальний розмір задає компонування
        widget.pack(fill="both", expand=True)
        self.fig.text(0.5, 0.5, hint, ha="center", va="center", color=GRAY)

    def draw(self, fn):
        self.fig.clear()
        try:
            fn(self.fig)
        except Exception:                                       # помилка одного графіка не має валити програму
            traceback.print_exc()
        self.canvas.draw_idle()


class Cards(ttk.Frame):
    """Ряд карток із великими числами (KPI)."""

    def __init__(self, parent, titles):
        super().__init__(parent)
        self.v = {}
        for i, (key, title) in enumerate(titles):
            box = ttk.LabelFrame(self, text=title)
            box.grid(row=0, column=i, sticky="nsew", padx=4)
            self.columnconfigure(i, weight=1)
            self.v[key] = tk.StringVar(value="—")
            ttk.Label(box, textvariable=self.v[key], style="Big.TLabel").pack(padx=10, pady=(2, 6))

    def set(self, **kw):
        for k, v in kw.items():
            self.v[k].set(v)


def make_text(parent, height=10, width=40) -> tk.Text:
    t = tk.Text(parent, wrap="word", height=height, width=width, relief="solid", bd=1, padx=8, pady=6,
                bg="white", highlightthickness=0)
    base = tkfont.nametofont("TkTextFont")
    bold = base.copy()
    bold.configure(weight="bold")
    t._bold = bold                                              # тримаємо посилання, щоб шрифт не зник
    t.tag_configure("h", font=bold, foreground=BLUE, spacing1=6)
    t.tag_configure("good", foreground=GREEN)
    t.tag_configure("bad", foreground=RED)
    t.tag_configure("muted", foreground=GRAY)
    t.configure(state="disabled")
    return t


def put(t: tk.Text, parts):
    t.configure(state="normal")
    t.delete("1.0", "end")
    for text, tag in parts:
        t.insert("end", text, tag)
    t.configure(state="disabled")


class Table(ttk.Frame):
    """Таблиця (Treeview) із вертикальною прокруткою."""

    def __init__(self, parent, cols, height=7):
        super().__init__(parent)
        self.tv = ttk.Treeview(self, columns=[c[0] for c in cols], show="headings", height=height)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tv.pack(side="left", fill="both", expand=True)
        for key, title, width in cols:
            self.tv.heading(key, text=title)
            self.tv.column(key, width=width, anchor="center")
        self.tv.tag_configure("bad", foreground=RED)
        self.tv.tag_configure("best", background="#DCFCE7")


def make_table(parent, cols, height=7) -> Table:
    return Table(parent, cols, height)


def fill(table: Table, rows, tags=None):
    tv = table.tv
    tv.delete(*tv.get_children())
    for i, r in enumerate(rows):
        tv.insert("", "end", values=r, tags=(tags[i],) if tags and tags[i] else ())


def add_row(parent, r, text, widget):
    ttk.Label(parent, text=text).grid(row=r, column=0, sticky="w", padx=(8, 4), pady=3)
    widget.grid(row=r, column=1, sticky="ew", padx=(4, 8), pady=3)


def spin(parent, var, lo, hi, step=1, width=9):
    return ttk.Spinbox(parent, textvariable=var, from_=lo, to=hi, increment=step, width=width)


# ================================================================= графіки
def plot_data(fig, log, an):
    (a1, a2), (a3, a4) = fig.subplots(2, 2)
    a1.bar(range(HOURS), log.hourly_per_day(), color=BLUE)
    a1.set_xticks(range(HOURS))
    a1.set_xticklabels([str(8 + i) for i in range(HOURS)])
    a1.set(title="Виклики за годинами доби", xlabel="година початку", ylabel="викликів/день")
    mon = log.monthly()
    x = mon["month"].to_numpy()
    a2.bar(x, mon["calls_day"], color=GREEN)
    a2.plot(x, an["intercept"] + an["slope"] * x, color=RED, lw=2, label=f"тренд {an['slope']:+.1f}/міс, R²={an['r2']:.2f}")
    a2.set_xticks(x)
    a2.set(title="Зростання навантаження за роком", xlabel="місяць", ylabel="викликів/день")
    a2.legend()
    s = log.df["service_length"].to_numpy(float)
    xs = np.linspace(0, 1500, 200)
    a3.hist(s, bins=50, range=(0, 1500), density=True, color=BLUE, alpha=0.6, label="дані")
    a3.plot(xs, np.exp(-xs / s.mean()) / s.mean(), color=RED, lw=2, label="експоненційний закон")
    a3.set(title="Тривалість розмови", xlabel="с", ylabel="щільність")
    a3.legend()
    a4.bar(x, mon["sla_pct"], color=[GREEN if v >= 90 else ORANGE for v in mon["sla_pct"]])
    a4.set_ylim(70, 100)
    a4.axhline(90, ls=":", color=GRAY)
    a4.set_xticks(x)
    a4.set(title="SLA та очікування за місяцями", xlabel="місяць", ylabel="SLA, %")
    b = a4.twinx()
    b.plot(x, mon["wait_mean"], "o-", color="#7C3AED")
    b.set_ylabel("сер. очікування, с")
    b.grid(False)


def plot_result(fig, res, ref, ref_label):
    a1, a2, a3 = fig.subplots(1, 3)
    w, bins = res.waits, np.linspace(0, 300, 31)
    pos = w[w > 0]
    a1.hist(pos, bins=bins, weights=np.full(len(pos), 100 / len(w)), color=BLUE, alpha=0.6, label="імітація")
    if ref is not None:
        r = ref[ref > 0]
        a1.hist(r, bins=bins, weights=np.full(len(r), 100 / len(ref)), histtype="step", color=RED, lw=2, label=ref_label)
    a1.axvline(res.cfg.sla, ls="--", color=GRAY)
    a1.set(title="Час очікування (тих, хто чекав)", xlabel="с", ylabel="% усіх викликів")
    a1.legend()
    t, q = 8 + sim.GRID / 3600, res.queue
    a2.fill_between(t, np.percentile(q, 10, axis=0), np.percentile(q, 90, axis=0), color=ORANGE, alpha=0.25, label="10–90%")
    a2.plot(t, q.mean(axis=0), color=ORANGE, lw=2, label="середня")
    a2.set(title="Довжина черги протягом дня", xlabel="година", ylabel="клієнтів у черзі")
    a2.legend()
    h = res.hourly
    a3.bar(h["hour"], h["wait_mean"], color=BLUE, alpha=0.8)
    a3.set_xticks(range(HOURS))
    a3.set_xticklabels([str(8 + i) for i in range(HOURS)])
    a3.set(title="Очікування і SLA за годинами", xlabel="година", ylabel="сер. очікування, с")
    b = a3.twinx()
    b.plot(h["hour"], h["sla_pct"], "o-", color=GREEN)
    b.set_ylabel("SLA, %")
    b.grid(False)


def plot_sweep(fig, df, target):
    a, b = fig.subplots(1, 2)
    n = df["agents"]
    a.errorbar(n, df["sla"], yerr=[df["sla"] - df["sla_lo"], df["sla_hi"] - df["sla"]], fmt="o-", color=BLUE, capsize=3,
               label="імітація (95% ЦІ)")
    a.plot(n, df["sla_erl"], "s--", color=GREEN, label="Erlang C")
    a.axhline(target, ls=":", color=RED, label=f"ціль {target:.0f}%")
    ok = df[df["sla"] >= target]
    if len(ok):
        a.axvline(ok["agents"].min(), color=RED, alpha=0.25, lw=10)
    a.set(title="SLA залежно від кількості операторів", xlabel="операторів", ylabel="SLA, %")
    a.legend(loc="lower right")
    b.bar(n, df["util"], color=ORANGE, alpha=0.8)
    b.set(title="Завантаження та очікування", xlabel="операторів", ylabel="завантаження, %")
    c = b.twinx()
    c.plot(n, df["wait"], "o-", color=BLUE)
    c.set_ylabel("сер. очікування, с")
    c.grid(False)


def plot_validation(fig, df):
    a, b = fig.subplots(1, 2)
    for ax, key, title, unit in ((a, "sla", "SLA за місяцями", "%"), (b, "wait", "Середнє очікування за місяцями", "с")):
        sm, sd, act, x = df[f"{key}_sim"], df[f"{key}_sd"], df[f"{key}_act"], df["month"]
        ax.fill_between(x, sm - 1.96 * sd, sm + 1.96 * sd, color=BLUE, alpha=0.2, label="імітація ±1,96σ")
        ax.plot(x, sm, "o-", color=BLUE, label="імітація")
        ax.plot(x, act, "D", color=RED, label="факт (журнал)")
        ax.set_xticks(x)
        ax.set(title=title, xlabel="місяць", ylabel=unit)
        ax.legend()


# ================================================================= вкладка «Дані»
class DataTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Button(top, text="Відкрити журнал (CSV / ZIP)…", command=app.open_dialog).pack(side="left")
        self.src = ttk.Label(top, text="Файл не завантажено", foreground=GRAY)
        self.src.pack(side="left", padx=10)
        self.cards = Cards(self, [("calls", "ВИКЛИКІВ"), ("days", "РОБОЧИХ ДНІВ"), ("cpd", "ВИКЛИКІВ / ДЕНЬ"),
                                  ("sla", "SLA ВИКОНАНО"), ("wait", "СЕР. ОЧІКУВАННЯ"), ("agents", "ОПЕРАТОРІВ")])
        self.cards.pack(fill="x", padx=10)
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=10, pady=8)
        self.text = make_text(body, width=52)
        self.text.pack(side="right", fill="y", padx=(8, 0))
        self.chart = Chart(body, (8, 6))
        self.chart.pack(side="left", fill="both", expand=True)

    def show(self, log, an):
        k = log.kpis()
        self.src.configure(text=f"Джерело: {log.source}")
        self.cards.set(calls=f"{k['calls']:,}".replace(",", " "), days=str(k["days"]), cpd=f"{k['calls_day']:.0f}",
                       sla=f"{k['sla_pct']:.1f}%", wait=f"{k['wait_mean']:.1f} с", agents=str(log.agents))
        self.chart.draw(lambda fig: plot_data(fig, log, an))
        parts = []
        for title, text in findings(log, an):
            parts += [(title + "\n", "h"), (text + "\n", "")]
        put(self.text, parts)


# ================================================================= вкладка «Симуляція»
class SimTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app, self.month = app, None
        left = ttk.LabelFrame(self, text="Параметри моделі")
        left.pack(side="left", fill="y", padx=10, pady=8)
        self.profile = tk.StringVar(value="Весь рік")
        self.load = tk.DoubleVar(value=100)
        self.agents, self.mean, self.sla = tk.IntVar(value=4), tk.DoubleVar(value=300), tk.DoubleVar(value=60)
        self.days, self.reps, self.seed = tk.IntVar(value=22), tk.IntVar(value=20), tk.IntVar(value=42)
        self.first = tk.BooleanVar(value=True)
        add_row(left, 0, "Потік викликів", ttk.Combobox(left, textvariable=self.profile, state="readonly", width=12,
                                                        values=["Весь рік"] + MONTHS))
        add_row(left, 1, "Навантаження, %", spin(left, self.load, 10, 500, 10))
        add_row(left, 2, "Операторів N", spin(left, self.agents, 1, 100))
        add_row(left, 3, "Розмова E[S], с", spin(left, self.mean, 10, 3600, 10))
        add_row(left, 4, "Поріг SLA, с", spin(left, self.sla, 5, 600, 5))
        add_row(left, 5, "Днів у прогоні", spin(left, self.days, 1, 365))
        add_row(left, 6, "Кількість прогонів", spin(left, self.reps, 2, 200))
        add_row(left, 7, "Seed", spin(left, self.seed, 0, 99999))
        ttk.Checkbutton(left, text="Перший дзвінок о 08:00:00\n(як у журналі)", variable=self.first).grid(
            row=8, column=0, columnspan=2, sticky="w", padx=8, pady=4)
        ttk.Button(left, text="Калібрувати за даними", command=self.calibrate).grid(row=9, column=0, columnspan=2, sticky="ew", padx=8, pady=(10, 3))
        ttk.Button(left, text="▶  Запустити (F5)", style="Accent.TButton", command=self.run).grid(row=10, column=0, columnspan=2, sticky="ew", padx=8, pady=3)
        ttk.Button(left, text="Зберегти в Excel…", command=self.save_excel).grid(row=11, column=0, columnspan=2, sticky="ew", padx=8, pady=3)

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=8)
        self.cards = Cards(right, [("sla", "SLA ВИКОНАНО"), ("wait", "СЕР. ОЧІКУВАННЯ"), ("p95", "P95 ОЧІКУВАННЯ"),
                                   ("util", "ЗАВАНТАЖЕННЯ"), ("qmax", "МАКС. ЧЕРГА")])
        self.cards.pack(fill="x", pady=(0, 6))
        self.text = make_text(right, height=9, width=60)
        self.text.pack(side="bottom", fill="x", pady=(6, 0))
        self.chart = Chart(right, (10, 3.8), "Задайте параметри й натисніть «Запустити» (F5)")
        self.chart.pack(fill="both", expand=True)

    def calibrate(self):
        log = self.app.log
        if log is None:
            return
        self.profile.set("Весь рік")
        self.agents.set(log.agents)
        self.mean.set(round(log.service_mean, 1))
        self.sla.set(log.sla)
        self.first.set(log.first_at_open)
        self.load.set(100)

    def config(self) -> sim.Config:
        log = self.app.log
        if log is None:
            raise ValueError("Спершу завантажте журнал дзвінків.")
        try:
            p = self.profile.get()
            month = None if p == "Весь рік" else MONTHS.index(p) + 1
            cfg = sim.Config(rates=tuple(log.rates(month)), agents=int(self.agents.get()),
                             service_mean=float(self.mean.get()), sla=float(self.sla.get()),
                             load=float(self.load.get()) / 100, first_at_open=self.first.get(),
                             days=int(self.days.get()), reps=int(self.reps.get()), seed=int(self.seed.get()))
        except (tk.TclError, ValueError):
            raise ValueError("Усі параметри мають бути коректними числами.")
        if not (1 <= cfg.agents <= 100 and cfg.service_mean > 0 and cfg.sla > 0 and cfg.days >= 1 and cfg.reps >= 2
                and 0.1 <= cfg.load <= 5):
            raise ValueError("Перевірте діапазони: операторів 1–100, навантаження 10–500%, прогонів ≥ 2, днів ≥ 1.")
        self.month = month
        return cfg

    def run(self):
        try:
            cfg = self.config()
        except ValueError as e:
            messagebox.showerror("Параметри", str(e))
            return
        self.app.run_bg(lambda p, s: sim.run(cfg, p, s), self.done, "Симуляція…")

    def done(self, res: sim.Result):
        self.app.result = res
        s = res.summary
        self.cards.set(sla=f"{s['sla_pct']['mean']:.1f}%", wait=f"{s['wait_mean']['mean']:.1f} с", p95=f"{s['p95']['mean']:.0f} с",
                       util=f"{s['util_pct']['mean']:.0f}%", qmax=f"{s['q_max']['mean']:.1f}")
        ref = self.app.log.sub(self.month)["wait_length"].to_numpy(float)
        label = "факт, " + (MONTHS[self.month - 1].lower() if self.month else "весь рік")
        self.chart.draw(lambda fig: plot_result(fig, res, ref, label))
        put(self.text, self.summary_text(res))
        self.app.status.set(f"Симуляцію завершено за {res.elapsed:.1f} с")

    def summary_text(self, res):
        cfg, s, target = res.cfg, res.summary, self.app.target.get()
        sla, u, e = s["sla_pct"], s["util_pct"]["mean"], res.erlang
        ok = sla["mean"] >= target
        level = "критично високе" if u > 90 else "високе" if u > 80 else "помірне" if u > 60 else "низьке"
        lam = np.asarray(cfg.rates) * cfg.load
        need = max(sim.min_agents(float(r), cfg.service_mean, target / 100, cfg.sla) for r in lam)
        wq = "∞ (система нестійка)" if math.isinf(e["wq"]) else f"{e['wq']:.1f} с"
        return [("Результат\n", "h"),
                (f"SLA (очікування ≤ {cfg.sla:.0f} с): {sla['mean']:.1f}% (95% ЦІ {sla['lo']:.1f}–{sla['hi']:.1f}). ", ""),
                (f"Ціль {target:.0f}% {'досягається' if ok else 'НЕ досягається'}.\n", "good" if ok else "bad"),
                (f"Середнє очікування {s['wait_mean']['mean']:.1f} с; у черзі опиняється {s['waited_pct']['mean']:.1f}% викликів; "
                 f"95% клієнтів чекають не довше {s['p95']['mean']:.0f} с. Завантаження операторів {u:.0f}% — {level}.\n", ""),
                ("Перевірка за формулою Erlang C\n", "h"),
                (f"Теорія: SLA ≈ {e['sla'] * 100:.1f}%, середнє очікування ≈ {wq}. "
                 f"Імітація: {sla['mean']:.1f}% і {s['wait_mean']['mean']:.1f} с.\n", ""),
                (f"Для SLA ≥ {target:.0f}% у пікову годину потрібно щонайменше {need} операторів (зараз {cfg.agents}).\n",
                 "bad" if need > cfg.agents else "good")]

    def save_excel(self):
        res = self.app.result
        if res is None:
            messagebox.showinfo("Експорт", "Спершу запустіть симуляцію.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", initialfile="rezultaty.xlsx", filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        c = res.cfg
        params = pd.DataFrame({"Параметр": ["Операторів", "Середня розмова, с", "Поріг SLA, с", "Навантаження", "Днів у прогоні", "Прогонів", "Seed"],
                               "Значення": [c.agents, c.service_mean, c.sla, c.load, c.days, c.reps, c.seed]})
        kpi = pd.DataFrame([{"Показник": KPI_NAMES[k], "Середнє": round(v["mean"], 3), "95% ЦІ від": round(v["lo"], 3),
                             "95% ЦІ до": round(v["hi"], 3)} for k, v in res.summary.items()])
        hourly = res.hourly.rename(columns={"hour": "Година (0 = 08:00)", "calls_day": "Викликів/день",
                                            "wait_mean": "Сер. очікування, с", "sla_pct": "SLA, %"}).round(3)
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                params.to_excel(xw, sheet_name="Параметри", index=False)
                kpi.to_excel(xw, sheet_name="KPI", index=False)
                hourly.to_excel(xw, sheet_name="За годинами", index=False)
            self.app.status.set(f"Збережено: {path}")
        except Exception as e:
            messagebox.showerror("Експорт", str(e))


# ================================================================= вкладка «Підбір штату»
class SweepTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=10, pady=8)
        self.lo, self.hi, self.reps = tk.IntVar(value=2), tk.IntVar(value=9), tk.IntVar(value=10)
        for text, w in (("Операторів від", spin(bar, self.lo, 1, 60, width=5)), ("до", spin(bar, self.hi, 1, 60, width=5)),
                        ("Цільовий SLA, %", spin(bar, app.target, 50, 99.9, 1, width=6)), ("Прогонів", spin(bar, self.reps, 2, 100, width=5))):
            ttk.Label(bar, text=text).pack(side="left", padx=(10, 3))
            w.pack(side="left")
        ttk.Button(bar, text="▶  Виконати підбір", style="Accent.TButton", command=self.run).pack(side="left", padx=16)
        ttk.Label(self, foreground=GRAY, wraplength=1100, justify="left",
                  text="Для кожної кількості операторів виконується серія симуляцій (параметри беруться з вкладки «Симуляція»). "
                       "Обирається мінімальний штат, що забезпечує цільовий SLA; результат порівнюється з формулою Erlang C.").pack(fill="x", padx=12)
        low = ttk.Frame(self)
        low.pack(side="bottom", fill="x", padx=10, pady=8)
        self.text = make_text(low, height=7, width=50)
        self.text.pack(side="right", fill="y", padx=(8, 0))
        self.table = make_table(low, [("n", "Операторів", 90), ("u", "Завантаж., %", 100), ("w", "Очікування, с", 110),
                                      ("s", "SLA, % (імітація)", 120), ("e", "SLA, % (Erlang C)", 120), ("p", "P95, с", 80)])
        self.table.pack(side="left", fill="both", expand=True)
        self.chart = Chart(self, (10, 4))
        self.chart.pack(fill="both", expand=True, padx=10)

    def run(self):
        try:
            cfg = self.app.tabs["sim"].config()
            lo, hi, reps, target = int(self.lo.get()), int(self.hi.get()), int(self.reps.get()), float(self.app.target.get())
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Параметри", str(e))
            return
        if hi < lo or hi - lo > 30 or reps < 2:
            messagebox.showerror("Параметри", "Діапазон має бути впорядкованим (не ширше 30 значень), прогонів ≥ 2.")
            return
        cfg = sim.replace(cfg, reps=reps)
        self.app.run_bg(lambda p, s: sim.sweep(cfg, range(lo, hi + 1), p, s), lambda df: self.done(df, target), "Підбір штату…")

    def done(self, df, target):
        self.chart.draw(lambda fig: plot_sweep(fig, df, target))
        ok = df[df["sla"] >= target]
        best = int(ok["agents"].min()) if len(ok) else None
        fill(self.table, [(int(r.agents), f"{r.util:.0f}", f"{r.wait:.1f}", f"{r.sla:.1f}", f"{r.sla_erl:.1f}", f"{r.p95:.0f}")
                          for r in df.itertuples()],
             ["best" if r.agents == best else ("" if r.sla >= target else "bad") for r in df.itertuples()])
        if best is None:
            parts = [("Результат\n", "h"), (f"У цьому діапазоні SLA {target:.0f}% не досягається — розширте діапазон.\n", "bad")]
        else:
            r = df[df["agents"] == best].iloc[0]
            parts = [("Результат\n", "h"), (f"Мінімальний штат для SLA ≥ {target:.0f}%: {best} операторів ", "good"),
                     (f"(SLA {r['sla']:.1f}%, завантаження {r['util']:.0f}%, середнє очікування {r['wait']:.1f} с).\n", "")]
        parts.append(("Зелений рядок — мінімальний достатній штат, червоні — SLA нижчий за ціль.", "muted"))
        put(self.text, parts)


# ================================================================= вкладка «Валідація»
class ValidationTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=10, pady=8)
        self.reps = tk.IntVar(value=8)
        ttk.Label(bar, text="Прогонів на місяць").pack(side="left")
        spin(bar, self.reps, 3, 100, width=5).pack(side="left", padx=6)
        ttk.Button(bar, text="▶  Виконати валідацію", style="Accent.TButton", command=self.run).pack(side="left", padx=12)
        ttk.Label(self, foreground=GRAY, wraplength=1100, justify="left",
                  text="Для кожного місяця журналу модель запускається з параметрами, відкаліброваними за даними цього місяця "
                       "(потік λ(t), кількість операторів, середня тривалість розмови), і порівнюється з фактичними показниками (у таблиці: факт → модель).").pack(fill="x", padx=12)
        low = ttk.Frame(self)
        low.pack(side="bottom", fill="x", padx=10, pady=8)
        self.text = make_text(low, height=8, width=58)
        self.text.pack(side="right", fill="y", padx=(8, 0))
        self.table = make_table(low, [("m", "Місяць", 90), ("c", "Викликів/день", 120), ("s", "SLA, %", 110),
                                      ("w", "Очікування, с", 120), ("u", "Завантаження, %", 130)], height=8)
        self.table.pack(side="left", fill="both", expand=True)
        self.chart = Chart(self, (10, 3.6), "Натисніть «Виконати валідацію»")
        self.chart.pack(fill="both", expand=True, padx=10)

    def run(self):
        log = self.app.log
        if log is None:
            messagebox.showinfo("Валідація", "Спершу завантажте журнал дзвінків.")
            return
        try:
            cfg = sim.config_from_log(log, reps=int(self.reps.get()))
        except (tk.TclError, ValueError):
            messagebox.showerror("Параметри", "Кількість прогонів має бути числом.")
            return
        self.app.run_bg(lambda p, s: sim.validate(log, cfg, p, s), self.done, "Валідація моделі…")

    def done(self, df):
        self.chart.draw(lambda fig: plot_validation(fig, df))
        fill(self.table, [(MONTHS[int(r.month) - 1], f"{r.calls_act:.0f} → {r.calls_sim:.0f}", f"{r.sla_act:.1f} → {r.sla_sim:.1f}",
                           f"{r.wait_act:.1f} → {r.wait_sim:.1f}", f"{r.util_act:.0f} → {r.util_sim:.0f}") for r in df.itertuples()])
        m = {k: sim.mape(df[f"{k}_act"], df[f"{k}_sim"]) for k in ("calls", "sla", "util", "wait")}
        inside = int(((df["sla_act"] >= df["sla_sim"] - 1.96 * df["sla_sd"]) & (df["sla_act"] <= df["sla_sim"] + 1.96 * df["sla_sd"])).sum())
        good = max(m["calls"], m["sla"], m["util"]) < 10
        put(self.text, [("Похибка моделі (MAPE)\n", "h"),
                        (f"викликів/день {m['calls']:.1f}%, SLA {m['sla']:.1f}%, завантаження {m['util']:.1f}%, "
                         f"середнє очікування {m['wait']:.1f}%.\n", ""),
                        (f"Фактичний SLA потрапляє в інтервал ±1,96σ імітації у {inside} з {len(df)} місяців.\n", ""),
                        ("Висновок\n", "h"),
                        ("Модель адекватно відтворює систему." if good else "Є суттєві відхилення — уточніть параметри.", "good" if good else "bad"),
                        ("\nMAPE < 10% — добре, 10–20% — прийнятно.", "muted")])
        self.app.status.set("Валідацію завершено")


# ================================================================= головне вікно
class App(tk.Tk):
    def __init__(self, path: str | None = None):
        super().__init__()
        self.title("Імітаційне моделювання контакт-центру")
        self.geometry("1280x820")
        self.minsize(1050, 680)
        self.configure(background=BG)
        self._setup_style()
        self.log, self.result = None, None
        self.target = tk.DoubleVar(value=90.0)                  # цільовий SLA, спільний для вкладок
        self.status = tk.StringVar(value="Готово")
        self.q, self.stop, self.busy = queue.Queue(), threading.Event(), False

        bar = ttk.Frame(self, style="Status.TFrame")
        bar.pack(side="bottom", fill="x")
        ttk.Label(bar, textvariable=self.status, style="Status.TLabel").pack(side="left", padx=12, pady=6)
        self.cancel_btn = ttk.Button(bar, text="Скасувати", style="Danger.TButton", command=self.stop.set, state="disabled")
        self.cancel_btn.pack(side="right", padx=10, pady=5)
        self.pb = ttk.Progressbar(bar, length=220)
        self.pb.pack(side="right", pady=5)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.tabs = {}
        for key, title, cls in (("data", "Дані", DataTab), ("sim", "Симуляція", SimTab),
                                ("sweep", "Підбір штату", SweepTab), ("val", "Валідація", ValidationTab)):
            self.tabs[key] = cls(nb, self)
            nb.add(self.tabs[key], text=f"  {title}  ")
        self.bind("<F5>", lambda e: self.tabs["sim"].run())
        self.bind("<Control-o>", lambda e: self.open_dialog())
        if path and os.path.exists(path):
            self.after(200, lambda: self.load(path))

    # ---- зовнішній вигляд: білий фон, синій акцент замість стандартної сірої теми
    def _setup_style(self):
        big = tkfont.nametofont("TkDefaultFont").copy()
        big.configure(size=16, weight="bold")
        self._big = big
        base = tkfont.nametofont("TkDefaultFont").copy()
        base.configure(size=10)
        self.option_add("*Font", base)

        st = ttk.Style(self)
        st.theme_use("clam")

        # --- базові контейнери
        st.configure(".", background=BG, foreground=TEXT, borderwidth=0, focuscolor=ACCENT)
        st.configure("TFrame", background=BG)
        st.configure("TLabel", background=BG, foreground=TEXT)
        st.configure("Big.TLabel", background=BG, foreground=ACCENT_DARK, font=big)
        st.configure("Status.TFrame", background=PANEL)
        st.configure("Status.TLabel", background=PANEL, foreground=MUTED2)

        # --- картки показників (LabelFrame)
        st.configure("TLabelframe", background=BG, bordercolor=BORDER, lightcolor=BG, darkcolor=BG,
                     borderwidth=1, relief="solid")
        st.configure("TLabelframe.Label", background=BG, foreground=MUTED2, font=(base.actual("family"), 9, "bold"))

        # --- вкладки
        st.configure("TNotebook", background=BG, bordercolor=BORDER, tabmargins=(4, 6, 4, 0))
        st.configure("TNotebook.Tab", background=PANEL, foreground=MUTED2, padding=(18, 9), borderwidth=0)
        st.map("TNotebook.Tab", background=[("selected", BG)], foreground=[("selected", ACCENT_DARK)],
               font=[("selected", (base.actual("family"), 10, "bold"))])

        # --- звичайні кнопки (світлі, з межею)
        st.configure("TButton", background=BG, foreground=TEXT, bordercolor=BORDER, lightcolor=BG, darkcolor=BG,
                     borderwidth=1, relief="solid", padding=(12, 7), focusthickness=0, focuscolor="none")
        st.map("TButton", background=[("disabled", PANEL), ("pressed", PANEL), ("active", PANEL)],
               foreground=[("disabled", "#B9C2CE")], bordercolor=[("active", ACCENT)])

        # --- акцентна кнопка (головна дія: «Запустити», «Виконати…»)
        st.configure("Accent.TButton", background=ACCENT, foreground="white", bordercolor=ACCENT,
                     lightcolor=ACCENT, darkcolor=ACCENT, borderwidth=0, relief="flat", padding=(14, 9),
                     font=(base.actual("family"), 10, "bold"), focusthickness=0, focuscolor="none")
        st.map("Accent.TButton", background=[("disabled", "#93B4F3"), ("pressed", ACCENT_DARK), ("active", ACCENT_DARK)],
               foreground=[("disabled", "white")])

        # --- кнопка небезпечної дії («Скасувати»)
        st.configure("Danger.TButton", background=BG, foreground=DANGER, bordercolor=DANGER, lightcolor=BG, darkcolor=BG,
                     borderwidth=1, relief="solid", padding=(12, 6), focusthickness=0, focuscolor="none")
        st.map("Danger.TButton", background=[("disabled", BG), ("active", "#FEF2F2")],
               foreground=[("disabled", "#D9B7B7")], bordercolor=[("disabled", BORDER)])

        # --- поля вводу
        for name in ("TEntry", "TSpinbox", "TCombobox"):
            st.configure(name, fieldbackground="white", background="white", foreground=TEXT,
                        bordercolor=BORDER, lightcolor="white", darkcolor="white", arrowcolor=MUTED2,
                        borderwidth=1, relief="solid", padding=4)
            st.map(name, bordercolor=[("focus", ACCENT)], fieldbackground=[("disabled", PANEL)])
        st.map("TCombobox", selectbackground=[("readonly", "white")], selectforeground=[("readonly", TEXT)],
               fieldbackground=[("readonly", "white")])

        # --- прапорці
        st.configure("TCheckbutton", background=BG, foreground=TEXT, focuscolor="none")
        st.map("TCheckbutton", indicatorcolor=[("selected", ACCENT)])

        # --- таблиця (Treeview)
        st.configure("Treeview", background="white", fieldbackground="white", foreground=TEXT,
                     bordercolor=BORDER, borderwidth=1, rowheight=24)
        st.configure("Treeview.Heading", background=PANEL, foreground=TEXT, borderwidth=0,
                     font=(base.actual("family"), 9, "bold"), relief="flat")
        st.map("Treeview.Heading", background=[("active", PANEL)])
        st.map("Treeview", background=[("selected", ACCENT_LIGHT)], foreground=[("selected", TEXT)])

        # --- смуга прогресу та прокрутка
        st.configure("TProgressbar", background=ACCENT, troughcolor=PANEL, bordercolor=PANEL, lightcolor=ACCENT, darkcolor=ACCENT)
        st.configure("TScrollbar", background=PANEL, troughcolor=BG, bordercolor=BG, arrowcolor=MUTED2, gripcount=0)
        st.map("TScrollbar", background=[("active", BORDER)])

        st.configure("TPanedwindow", background=BG)

    # ---- дані
    def open_dialog(self):
        path = filedialog.askopenfilename(title="Журнал дзвінків", filetypes=[("CSV або ZIP", "*.csv *.zip"), ("Усі файли", "*.*")])
        if path:
            self.load(path)

    def load(self, path):
        def work(progress, should_stop):
            progress(0.3, "Читання файлу…")
            log = load_csv(path)
            progress(0.7, "Аналіз даних…")
            return log, analyze(log)

        self.run_bg(work, self.loaded, "Завантаження даних…")

    def loaded(self, res):
        self.log, an = res
        self.tabs["data"].show(self.log, an)
        self.tabs["sim"].calibrate()
        self.status.set(f"Завантажено {len(self.log.df):,} викликів; оцінка кількості операторів: {self.log.agents}".replace(",", " "))

    # ---- фонове виконання: обчислення в потоці, інтерфейс не зависає
    def run_bg(self, fn, done, text="Виконання…"):
        if self.busy:
            return
        self.busy = True
        self.stop.clear()
        self.status.set(text)
        self.pb["value"] = 0
        self.cancel_btn.state(["!disabled"])

        def work():
            try:
                self.q.put(("ok", fn(lambda f, m="": self.q.put(("p", f, m)), self.stop.is_set)))
            except sim.Cancelled:
                self.q.put(("cancel", None))
            except Exception as e:                              # noqa: BLE001
                traceback.print_exc()
                self.q.put(("err", e))

        threading.Thread(target=work, daemon=True).start()
        self.after(50, lambda: self.poll(done))

    def poll(self, done):
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == "p":
                    self.pb["value"] = m[1] * 100
                    if m[2]:
                        self.status.set(m[2])
                    continue
                self.busy = False
                self.pb["value"] = 0
                self.cancel_btn.state(["disabled"])
                if m[0] == "ok":
                    try:
                        done(m[1])
                    except Exception as e:                      # noqa: BLE001
                        traceback.print_exc()
                        messagebox.showerror("Помилка", f"{type(e).__name__}: {e}")
                elif m[0] == "cancel":
                    self.status.set("Виконання перервано")
                else:
                    messagebox.showerror("Помилка", f"{type(m[1]).__name__}: {m[1]}")
                return
        except queue.Empty:
            pass
        self.after(50, lambda: self.poll(done))


def main(argv=None):
    path = argv[0] if argv else DEFAULT_DATA
    App(path).mainloop()
