from src.task2.bpr import fit_bpr
import pandas as pd

df = pd.read_csv("assignment1_2026/tasks_1_2_train.csv", nrows=20000)
art = fit_bpr(
    df,
    {"factors": 16, "iterations": 5, "use_gpu": True, "show_progress": False},
    seed=0,
)
print("ok", art.user_factors.shape, art.item_factors.dtype, art.user_factors.dtype)
print(float(art.user_factors[0, 0]))
