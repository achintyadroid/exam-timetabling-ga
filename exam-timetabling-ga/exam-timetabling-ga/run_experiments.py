"""Run all experiments and save results/ and screenshots/ for the README and report.
Usage:  python run_experiments.py
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import timetabling as T

SEEDS = range(3)          # change to range(10) for the final run
GENS = 600
PARAM_SEEDS = range(5)
PARAM_GENS = 300
os.makedirs("results", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

problem = T.Problem("data")
log = []
def say(msg=""):
    print(msg); log.append(msg)

say(f"Exams={problem.n}  Rooms={problem.n_rooms}  Slots={T.N_SLOTS}  "
    f"Students={int(problem.S.shape[0])}")

# ---- baselines ----
rand_mean, rand_best = T.random_baseline(problem, 1000)
greedy = T.greedy_baseline(problem)
greedy_bd = problem.breakdown(greedy)
say(f"Random baseline (1000 samples): mean={rand_mean:.1f}  best={rand_best:.1f}")
say(f"Greedy baseline: fitness={greedy_bd['fitness']:.1f}  hard violations={greedy_bd['hard_violations']:.0f}")

# ---- main runs: GA vs GA + local search (memetic variant) ----
runs = {"GA": [], "GA + local search": []}
best_overall = {}
for name, memetic in [("GA", False), ("GA + local search", True)]:
    for s in SEEDS:
        t0 = time.time()
        b, f, bh, mh = T.run_ga(problem, generations=GENS, memetic=memetic, seed=s)
        runs[name].append({"seed": s, "fitness": float(f), "best_hist": bh.tolist(),
                           "mean_hist": mh.tolist(), "time": time.time() - t0,
                           **problem.breakdown(b)})
        if name not in best_overall or f < best_overall[name][0]:
            best_overall[name] = (f, b)
        say(f"{name:18s} seed={s} fitness={f:9.2f} hard={problem.breakdown(b)['hard_violations']:.0f} "
            f"time={time.time() - t0:.1f}s")

summary = {"random_mean": rand_mean, "random_best": rand_best, "greedy": greedy_bd}
for name, r in runs.items():
    fs = np.array([x["fitness"] for x in r])
    summary[name] = {
        "mean": fs.mean(), "std": fs.std(), "best": fs.min(), "worst": fs.max(),
        "mean_time_s": float(np.mean([x["time"] for x in r])),
        "mean_hard": float(np.mean([x["hard_violations"] for x in r])),
        "mean_back_to_back": float(np.mean([x["back_to_back"] for x in r])),
        "mean_overload_day": float(np.mean([x["overload_day"] for x in r])),
        "zero_hard_runs": int(sum(x["hard_violations"] == 0 for x in r)),
    }
say(); say("=== Summary over %d runs (%d generations) ===" % (len(SEEDS), GENS))
for name in runs:
    s = summary[name]
    say(f"{name:18s} mean={s['mean']:.1f} ± {s['std']:.1f}  best={s['best']:.1f}  "
        f"hard-free runs={s['zero_hard_runs']}/{len(SEEDS)}  time={s['mean_time_s']:.1f}s")

# ---- parameter study (one factor at a time around the default) ----
default = dict(pop_size=100, pc=0.8, pm=0.03)
grid = {"pop_size": [20, 50, 100, 200], "pm": [0.005, 0.01, 0.03, 0.08], "pc": [0.5, 0.8, 0.95]}
param = {}
for key, values in grid.items():
    param[key] = {}
    for v in values:
        cfg = dict(default); cfg[key] = v
        fs = [T.run_ga(problem, generations=PARAM_GENS, seed=s, **cfg)[1] for s in PARAM_SEEDS]
        param[key][str(v)] = {"mean": float(np.mean(fs)), "std": float(np.std(fs))}
        say(f"param {key}={v}: mean={np.mean(fs):.1f} ± {np.std(fs):.1f}")

json.dump({"summary": summary, "runs": runs, "param_study": param},
          open("results/results.json", "w"), indent=1, default=float)

# ---------- figures ----------
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})

# 1) convergence
fig, ax = plt.subplots(figsize=(7, 4))
for name, c in [("GA", "tab:blue"), ("GA + local search", "tab:orange")]:
    h = np.array([x["best_hist"] for x in runs[name]])
    ax.plot(h.mean(0), color=c, label=name)
    ax.fill_between(range(h.shape[1]), h.min(0), h.max(0), color=c, alpha=0.15)
ax.set_yscale("log"); ax.set_xlabel("Generation"); ax.set_ylabel("Best fitness (log scale)")
ax.set_title("Convergence (mean and min-max band)"); ax.legend()
fig.tight_layout(); fig.savefig("screenshots/01_convergence.png"); plt.close(fig)

# 2) comparison bar
labels = ["Random\n(mean)", "Greedy", "GA", "GA + local\nsearch"]
vals = [rand_mean, greedy_bd["fitness"], summary["GA"]["mean"], summary["GA + local search"]["mean"]]
errs = [0, 0, summary["GA"]["std"], summary["GA + local search"]["std"]]
fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.bar(labels, vals, yerr=errs, color=["#999", "#c9a227", "tab:blue", "tab:orange"], capsize=4)
ax.set_yscale("log"); ax.set_ylabel("Fitness (log scale, lower is better)")
ax.set_title("Method comparison")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v:.0f}", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig("screenshots/02_method_comparison.png"); plt.close(fig)

# 3) parameter study
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, (key, title) in zip(axes, [("pop_size", "Population size"), ("pm", "Mutation rate"),
                                   ("pc", "Crossover rate")]):
    xs = list(param[key].keys()); m = [param[key][x]["mean"] for x in xs]
    sd = [param[key][x]["std"] for x in xs]
    ax.errorbar(range(len(xs)), m, yerr=sd, marker="o", capsize=4)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs); ax.set_title(title)
axes[0].set_ylabel("Final fitness (mean ± std, 5 runs)")
fig.tight_layout(); fig.savefig("screenshots/03_parameter_study.png"); plt.close(fig)

# 4) timetable heatmap of the best GA solution
f, best = best_overall["GA"]
grid_ = np.full((problem.n_rooms, T.N_SLOTS), -1)
for e, (s, r) in enumerate(best):
    grid_[r, s] = e
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.imshow(grid_ >= 0, cmap="Blues", aspect="auto", vmin=0, vmax=1.6)
for r in range(problem.n_rooms):
    for s in range(T.N_SLOTS):
        if grid_[r, s] >= 0:
            ax.text(s, r, problem.codes[grid_[r, s]], ha="center", va="center", fontsize=7)
ax.set_xticks(range(T.N_SLOTS))
ax.set_xticklabels([f"D{s // T.SLOTS_PER_DAY + 1}\nS{s % T.SLOTS_PER_DAY + 1}" for s in range(T.N_SLOTS)])
ax.set_yticks(range(problem.n_rooms))
ax.set_yticklabels([f"Room {r} ({problem.cap[r]})" for r in range(problem.n_rooms)])
ax.grid(False); ax.set_title(f"Best GA timetable (fitness {f:.1f})")
fig.tight_layout(); fig.savefig("screenshots/04_best_timetable.png"); plt.close(fig)

# 5) console output as an image
fig, ax = plt.subplots(figsize=(10, 0.22 * len(log) + 0.6)); ax.axis("off")
fig.patch.set_facecolor("#1e1e1e")
ax.text(0, 1, "\n".join(log), family="monospace", fontsize=7, color="#d4d4d4", va="top")
fig.savefig("screenshots/05_console_output.png", facecolor="#1e1e1e", bbox_inches="tight"); plt.close(fig)

np.savetxt("results/best_timetable_ga.csv",
           np.column_stack([np.arange(problem.n), best, problem.size]), fmt="%d", delimiter=",",
           header="exam_id,timeslot,room,size", comments="")
say("\nSaved results/ and screenshots/")