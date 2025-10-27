import os
import sys
import subprocess
import csv
import re

graphs = ['USA-road-d.NY.egr','amazon0601.egr','rmat16.sym.egr','in-2004.egr', 'coPapersDBLP.egr', 'soc-LiveJournal1.egr','internet.egr', 'delaunay_n22.egr']
graph_direct = "graphs/"
args = sys.argv
csv_file = f"{args[1]}.csv"
core = args[1]

with open(csv_file, "w", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    patterns = []
    pattern_dir = "patterns/"
    for pattern_file in os.listdir(pattern_dir):
        if pattern_file.endswith('.txt') and pattern_file.startswith(f"{core}"):
            base, exttension = os.path.splitext(pattern_file)
            patterns.append(base)

    patterns = sorted(patterns)
    print(f"length: {len(patterns)}")
    header_row = ['Graph  File']
    csv2row = ['Graph File']
    for subgraph in patterns:
        header_row.append(f"Fringe-SGC GPU {subgraph}")

    print(header_row)
    csv_writer.writerow(header_row) 
    for graph in graphs:
        row = [graph]
        info_c = [f"./src/info", f"graphs/{graph}"]
        info_result = subprocess.run(info_c, capture_output=True, text=True)
        vertices = re.search(r'nodes: (\d+)', info_result.stdout)
        vertices = int(vertices.group(1)) if vertices else None
        edges = re.search(r'edges: (\d+)', info_result.stdout)
        edges = int(edges.group(1)) if edges else None
        if vertices is None:
            print("There was an error in downloading graphs")
            sys.exit()

        gfc_total_r = 0.0
        gfc_total_p = 0.0
        gfc_total_g = 0.0

        for pattern in patterns:
            without_extension = os.path.splitext(graph)[0]
            pattern_text = ""
            filename = f"patterns/{pattern}.txt"
            with open(filename, 'r') as sub:
                first_line = sub.readline().strip()
                match = re.search(r'Nodes:\s*(\d+)', first_line)
                if match:
                    nodes = int(match.group(1))
                    pattern_text = str(nodes)
                    pairs = set()
                    for line in sub:
                        a, b = map(int, line.split())
                        pairs.add(tuple(sorted((a,b))))

                pattern_text = f"{nodes} " + " ".join(f"{a} {b}" for a, b in sorted(pairs))

            arguments = pattern_text.split()
            pre_command = ["./src/fringePreprocess"]
            pre_command.extend(arguments)
            result = subprocess.run(pre_command, capture_output=True, text=True,timeout=3600)

            gfc_command = [f"./src/fringeCount", f"graphs/{graph}", "motif.mo"]
            
            try:
                if (gfc_total_r != 'Timeout' and gfc_total_r < 3600): 
                    gfc_result = subprocess.run(gfc_command, capture_output=True,text=True,timeout=3600)
                    gfc_p_time = re.search(r'Pattern preprocessing time: (\d+\.\d+)', gfc_result.stdout)
                    gfc_g_time = re.search(r'Preprocessing graph: (\d+\.\d+)',gfc_result.stdout)
                    gfc_r_time = re.search(r'runtime: (\d+\.\d+)',gfc_result.stdout)
                    gfc_p_time = float(gfc_p_time.group(1)) if gfc_p_time else None
                    gfc_g_time = float(gfc_g_time.group(1)) if gfc_g_time else None
                    gfc_r_time = float(gfc_r_time.group(1)) if gfc_r_time else None
                    gfc_c = re.search(r'occurs (\d+)', gfc_result.stdout)
                    gfc_c = int(gfc_c.group(1)) if gfc_c else 0
                else:
                    gfc_total_r = "Timeout"

            except subprocess.TimeoutExpired:
                gfc_p_time = "Timeout"
                gfc_g_time = "Timeout"
                gfc_r_time = -1
                gfc_c = "Timeout"

            row.append(edges / gfc_r_time)
            
        print(row)
        csv_writer.writerow(row)
        print(f"{graph} finished")
print("done!")

