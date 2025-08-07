import subprocess
import os
import re
import glob

runs = ["vertex", "edge", "triangle", "wedge"] 
command = ["python3", "reproduce.py"]

for run in runs:
    current_command = command.copy() 
    current_command.append(run)
    subprocess.run(current_command, check=True)
    current_command = ["python3", "geometric_fringe.py", f"{run}.csv", f"geo_{run}.csv"]
    subprocess.run(current_command, check=True)
    current_command = ["gnuplot", "-c", "plot_fringe.gp", f"geo_{run}.csv"]
    subprocess.run(current_command, check=True)

fig3_runs = ["tails", "wedge", "tri_fringe"]

command.append("fig3")

for run in fig3_runs:
    files = glob.glob(f"patterns/{run}/fig3*")
    subprocess.run(["scp"] + files + ["patterns/"], check=True)
    subprocess.run(command, check=True)
    move_command = ["mv", f"fig3.csv", f"fig3_{run}.csv"]
    subprocess.run(move_command, check=True)
    geo_command = ["python3", "geometric_fringe.py", f"fig3_{run}.csv", f"geo_{run}.csv"]
    subprocess.run(geo_command, check=True)
    for f in glob.glob("patterns/fig3*"):
        os.remove(f)

    plot_command = ["gnuplot", "-c", "plot_fig3.gp", f"geo_{run}.csv"]
    subprocess.run(plot_command, check=True)


