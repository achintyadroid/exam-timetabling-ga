"""Exam timetabling: problem model, fitness function and Genetic Algorithm.

Chromosome : integer array of shape (n_exams, 2); row i = (timeslot, room) of exam i.
Fitness    : weighted penalty (lower is better, 0 = perfect).
"""
import os
import numpy as np
import pandas as pd

SLOTS_PER_DAY = 3
N_DAYS = 5
N_SLOTS = SLOTS_PER_DAY * N_DAYS

# Penalty weights
W_CLASH = 100.0     # HARD: student has two exams in the same slot (per student)
W_ROOM = 100.0      # HARD: two exams in same room and slot
W_CAP = 50.0        # HARD: exam larger than its room
W_ADJ = 2.0         # SOFT: student has back-to-back exams in the same day (per student)
W_DAY = 5.0         # SOFT: student has more than 2 exams on one day (per extra exam)
W_LATE = 1.0        # SOFT: big exams should be early in the exam period
COMPONENTS = ["clash", "room_conflict", "capacity", "back_to_back", "overload_day", "late_big"]
WEIGHTS = np.array([W_CLASH, W_ROOM, W_CAP, W_ADJ, W_DAY, W_LATE])


class Problem:
    def __init__(self, data_dir):
        exams = pd.read_csv(os.path.join(data_dir, "exams.csv"))
        rooms = pd.read_csv(os.path.join(data_dir, "rooms.csv"))
        enr = pd.read_csv(os.path.join(data_dir, "enrollments.csv"))
        self.n = len(exams)
        self.size = exams["size"].to_numpy()
        self.codes = exams["course_code"].tolist()
        self.cap = rooms["capacity"].to_numpy()
        self.n_rooms = len(rooms)
        n_students = enr["student_id"].max() + 1
        S = np.zeros((n_students, self.n))
        S[enr["student_id"], enr["exam_id"]] = 1
        self.S = S
        self.conflict = S.T @ S                      # students shared by exam i and j
        np.fill_diagonal(self.conflict, 0)
        self.triu = np.triu(np.ones((self.n, self.n), dtype=bool), 1)
        self.day_onehot = np.zeros((N_SLOTS, N_DAYS))
        self.day_onehot[np.arange(N_SLOTS), np.arange(N_SLOTS) // SLOTS_PER_DAY] = 1
        self.size_norm = self.size / self.size.max()
        # rooms that can physically hold each exam
        self.feasible_rooms = [np.where(self.cap >= s)[0] for s in self.size]

    # ---------- fitness ----------
    def components(self, ind):
        slot, room = ind[:, 0], ind[:, 1]
        same_slot = slot[:, None] == slot[None, :]
        clash = (self.conflict * same_slot * self.triu).sum()
        room_conf = (same_slot & (room[:, None] == room[None, :]) & self.triu).sum()
        capacity = (self.size > self.cap[room]).sum()
        day, pos = slot // SLOTS_PER_DAY, slot % SLOTS_PER_DAY
        adjacent = (day[:, None] == day[None, :]) & (np.abs(pos[:, None] - pos[None, :]) == 1)
        back_to_back = (self.conflict * adjacent * self.triu).sum()
        per_day = self.S @ self.day_onehot[slot]     # exams per student per day
        overload = np.maximum(per_day - 2, 0).sum()
        late_big = (self.size_norm * slot / (N_SLOTS - 1)).sum()
        return np.array([clash, room_conf, capacity, back_to_back, overload, late_big], float)

    def fitness(self, ind):
        return float(self.components(ind) @ WEIGHTS)

    def breakdown(self, ind):
        c = self.components(ind)
        d = dict(zip(COMPONENTS, c.tolist()))
        d["hard_violations"] = float(c[0] + c[1] + c[2])
        d["fitness"] = float(c @ WEIGHTS)
        return d

    # ---------- helpers ----------
    def random_gene(self, i, rng):
        return rng.integers(N_SLOTS), rng.choice(self.feasible_rooms[i])

    def random_individual(self, rng):
        return np.array([self.random_gene(i, rng) for i in range(self.n)])


# ---------- Genetic Algorithm ----------
def local_search(problem, ind, f, rng, moves):
    """Simple stochastic hill-climber (used only for the memetic variant)."""
    for _ in range(moves):
        i = rng.integers(problem.n)
        old = ind[i].copy()
        ind[i] = problem.random_gene(i, rng)
        nf = problem.fitness(ind)
        if nf < f:
            f = nf
        else:
            ind[i] = old
    return ind, f


def run_ga(problem, pop_size=100, generations=300, pc=0.8, pm=0.03, elite=5,
           tournament_k=3, memetic=False, ls_top=5, ls_moves=30, seed=0):
    rng = np.random.default_rng(seed)
    pop = np.array([problem.random_individual(rng) for _ in range(pop_size)])
    fit = np.array([problem.fitness(p) for p in pop])
    best_hist, mean_hist = [], []

    def select():
        idx = rng.integers(pop_size, size=tournament_k)
        return pop[idx[np.argmin(fit[idx])]]

    for _ in range(generations):
        order = np.argsort(fit)
        new_pop = [pop[i].copy() for i in order[:elite]]
        new_fit = [fit[i] for i in order[:elite]]
        while len(new_pop) < pop_size:
            p1, p2 = select(), select()
            if rng.random() < pc:                       # uniform crossover on genes
                mask = rng.random(problem.n) < 0.5
                c1 = np.where(mask[:, None], p1, p2)
                c2 = np.where(mask[:, None], p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()
            for c in (c1, c2):
                for i in np.where(rng.random(problem.n) < pm)[0]:   # gene mutation
                    if rng.random() < 0.5:
                        c[i, 0] = rng.integers(N_SLOTS)             # move to new slot
                    else:
                        c[i, 1] = rng.choice(problem.feasible_rooms[i])  # new room
                if rng.random() < 0.1:                              # swap mutation
                    a, b = rng.integers(problem.n, size=2)
                    c[[a, b], 0] = c[[b, a], 0]
                if len(new_pop) < pop_size:
                    new_pop.append(c)
                    new_fit.append(problem.fitness(c))
        pop, fit = np.array(new_pop), np.array(new_fit)
        if memetic:                                     # refine the best few individuals
            for i in np.argsort(fit)[:ls_top]:
                pop[i], fit[i] = local_search(problem, pop[i], fit[i], rng, ls_moves)
        best_hist.append(fit.min())
        mean_hist.append(fit.mean())
    b = int(np.argmin(fit))
    return pop[b], fit[b], np.array(best_hist), np.array(mean_hist)


# ---------- Baselines ----------
def random_baseline(problem, n_samples=1000, seed=0):
    rng = np.random.default_rng(seed)
    f = np.array([problem.fitness(problem.random_individual(rng)) for _ in range(n_samples)])
    return float(f.mean()), float(f.min())


def greedy_baseline(problem):
    """Largest-degree-first graph-colouring style heuristic."""
    degree = (problem.conflict > 0).sum(1)
    ind = np.zeros((problem.n, 2), int)
    placed = np.full(problem.n, -1)
    used = {}                                            # slot -> set of rooms in use
    for i in np.argsort(-degree):
        best = None
        for s in range(N_SLOTS):
            others = np.where(placed == s)[0]
            clash = problem.conflict[i, others].sum()
            free = [r for r in problem.feasible_rooms[i] if r not in used.get(s, set())]
            if not free:
                continue
            key = (clash, s)
            if best is None or key < best[0]:
                best = (key, s, min(free, key=lambda r: problem.cap[r]))
        _, s, r = best
        ind[i] = (s, r)
        placed[i] = s
        used.setdefault(s, set()).add(r)
    return ind