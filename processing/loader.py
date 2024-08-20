import csv
import json
import torch

career = {}
combine = {}
senior = {}
rookie = {}
training_data = []

with open('data/career_stats.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'Rk':
            name = row[1]
            
            g = int(row[10])
            rec = int(row[11]) / g
            yds = int(row[12]) / g
            td = int(row[14]) / g
            
            draft_pick = int(row[5])
            career[name] = [g, rec, yds, td, draft_pick]
            
with open('data/combine_stats.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] and row[0] != 'Year':
            name = row[1]
            height = float(row[4])
            weight = int(row[5])
            hand_size = float(row[6]) if row[6] else None
            arm_length = float(row[7]) if row[7] else None
            forty_time = float(row[8]) if row[8] else None
            vertical = float(row[10]) if row[10] else None
            broad = int(row[11]) if row[11] else None
            shuttle = float(row[12]) if row[12] else None
            three_cone = float(row[13]) if row[13] else None
            combine[name] = [height, weight, hand_size, arm_length, forty_time, vertical, broad, shuttle, three_cone]
            
            
with open('data/season_stats.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'Rk' and int(row[9]) + 1 == int(row[7]):
            name = row[1]
            g = int(row[11])
            catches = int(row[12]) / g
            yards = int(row[13]) / g
            tds = int(row[15]) / g
            senior[name] = [g, yards, catches, tds]
        
with open('data/rookie_stats.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'Rk':
            name = row[1]
            rec = int(row[10])
            yards = int(row[11])
            td = int(row[13])
            rookie[name] = [rec, yards, td]
            
def get_value_for_key(key, dict, length, name):
    if value := dict.get(key):
        return value
    return [None] * length
        
for key in rookie:
    career_stats = get_value_for_key(key, career, 5, 'career')
    senior_stats = get_value_for_key(key, senior, 4, 'senior')
    combine_stats = get_value_for_key(key, combine, 9, 'combine')
    
    stats = [*career_stats, *senior_stats, *combine_stats]
    if any(stats):
        datum = {'params': stats, 'results': rookie[key]}
        training_data.append(datum)
    else:
        print(f'no data for {key}')
        
with open('data/processed.json', 'w') as f: 
     f.write(json.dumps(training_data))
     

def load_data():
    with open('data/processed.json', 'r') as f:
        return json.load(f)