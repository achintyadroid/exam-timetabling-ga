"""Generate a synthetic university exam-timetabling dataset.

Students belong to one of 6 programmes. Each student sits 4 of their programme's
8 core exams plus 1-2 exams from a shared pool of 12 electives.

Outputs (in this folder):
  exams.csv        exam_id, course_code, size
  enrollments.csv  student_id, exam_id
  rooms.csv        room_id, capacity
Run:  python data/generate_dataset.py
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N_PROGRAMMES, CORE_PER_PROG, N_ELECTIVES = 6, 8, 12
STUDENTS_PER_PROG = 100
ROOM_CAPACITIES = [200, 200, 120, 120, 120, 80, 80, 80, 80, 50, 50, 50]

here = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(SEED)
n_core = N_PROGRAMMES * CORE_PER_PROG
n_exams = n_core + N_ELECTIVES
elective_ids = np.arange(n_core, n_exams)

rows, sid = [], 0
for p in range(N_PROGRAMMES):
    core = np.arange(p * CORE_PER_PROG, (p + 1) * CORE_PER_PROG)
    for _ in range(STUDENTS_PER_PROG):
        picks = list(rng.choice(core, 4, replace=False))
        picks += list(rng.choice(elective_ids, rng.integers(1, 3), replace=False))
        rows += [(sid, int(e)) for e in picks]
        sid += 1
enr = pd.DataFrame(rows, columns=["student_id", "exam_id"])

sizes = enr.groupby("exam_id").size().reindex(range(n_exams), fill_value=0)
assert sizes.max() <= max(ROOM_CAPACITIES), "an exam is larger than the biggest room"
exams = pd.DataFrame({"exam_id": range(n_exams),
                      "course_code": [f"CS{100 + i}" for i in range(n_exams)],
                      "size": sizes.values})
rooms = pd.DataFrame({"room_id": range(len(ROOM_CAPACITIES)), "capacity": ROOM_CAPACITIES})
exams.to_csv(os.path.join(here, "exams.csv"), index=False)
enr.to_csv(os.path.join(here, "enrollments.csv"), index=False)
rooms.to_csv(os.path.join(here, "rooms.csv"), index=False)
print(f"{n_exams} exams, {sid} students, {len(enr)} enrollments, "
      f"largest exam = {sizes.max()}, smallest = {sizes.min()}")