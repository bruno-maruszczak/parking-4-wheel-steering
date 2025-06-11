import json
import numpy as np


# Wczytaj plik z wynikami MPC
with open("data/out/mpc_results.json", "r") as f:
    mpc_results = json.load(f)

# Wczytaj plik z trajektorią referencyjną
with open("data/out/fourws_one_side_path.json", "r") as f:
    reference_path = json.load(f)

print(mpc_results['s'])

print(len(reference_path['x']))


# x = f(s,n)
# y = g(s,n)

# fourws_one_side_path.json
# x y contr_front contr_rear