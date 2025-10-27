import csv
import math
import sys
from itertools import zip_longest

bars = ['Fringe-SGC'] 

def geometric_mean(values):
    product = 1
    for value in values:
        product *= value
    #print(f"len: {len(values)}, product: {product}")
    return math.pow(product, 1 / len(values))

if len(sys.argv) < 3:
    print("usage: python geometric.py <csv_file> <output_name>")
    sys.exit()

csv_file = sys.argv[1]
output = sys.argv[2]
num_col = []
patterns = []
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    header = reader.fieldnames#next(reader)
    rows = list(reader)
    for i, col in enumerate(header):
        if i > 0:
            current_pattern = col.split(' ')[-1]
            if current_pattern not in patterns:
                patterns.append(current_pattern)
            data = []
            file.seek(0)
            next(reader)
            for row in reader:
                value = float(row[col]) 
                data.append(value)
            num_col.append(data)
    
    if not num_col:
        print("No numerical columns found in CSV file.")
        sys.exit()

    means = [geometric_mean(column) for column in num_col]

    for i, mean in enumerate(means):
        print(f"Geometric mean of '{header[i+1]}' : {mean}")
    
    with open(output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        row = ['Pattern']
        row += patterns
        writer.writerow(row)
        for i, bar in enumerate(bars):
            row = [bar] 
            for j in range(i, len(means), 1):
                row.append(means[j])
            writer.writerow(row)
  

    with open(output, 'r') as csvfile:
        reader = csv.reader(csvfile)
        dataF = list(reader)

    transposed = list(zip_longest(*dataF, fillvalue=''))

    with open(output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in transposed:
            writer.writerow(row)






