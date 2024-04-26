from Simulations import *
import sys
import os
import numpy as np

print("start")

output = """Select the simulation you want to run (separated by space):
            1. Varying N for Barnes-Hut
            2. Varying theta for Barnes-Hut
            3. Comparing Fast Multipole Method and Barnes-Hut
            4. Varying N for Fast Multipole Method
            5. Varying p for Fast Multipole Method
            6. Varying levels for Fast Multipole Method
            7. Exit
            """
try:
    simul_list = [int(i) for i in input(output).split(" ")]
except:
    print("End programme")
    sys.exit(0)

for simul in simul_list:
    if simul==1:
        bh_varying_n()
    if simul==2:
        bh_varying_theta()
    if simul==3:
        fmm_vs_bh()
    if simul==4:
        fmm_varying_n()
    if simul==5:
        fmm_varying_p()
    if simul==6:
        fmm_varying_levels()
    else:
        sys.exit(0)

