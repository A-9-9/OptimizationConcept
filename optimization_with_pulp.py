from pulp import *
import pandas as pd
# =============================================================================
# https://towardsdatascience.com/how-to-solve-optimization-problems-with-python-9088bf8d48e5
# How to Solve Optimization Problems with Python
# 
# Objective Function: Maximize Projected Points from our 9 Players.
# Constrains: 
    # Only buy a player a maximum of 1 times.
    # Own 2 point guards, 2 shooting guards, 2 small forwards, 2 power forwards, and 1 center.
    # Spend no more than $60,000.
    
        # Solve steps:
        # (1) Player constraints(maximum of 1 times)
        # (2) Maximize projected points
        # (3) Salary constraints
        # (4) Position constraints
        
# =============================================================================

# Initialize Dictionaries for Salaries, positions, project points
players = list(data['Nickname'])
salaries = dict(zip(players, data['Salary']))
positions = dict(zip(players, data['Position']))
project_points = dict(zip(players, data['FPPG']))

# Step (1)
# Constraints of target variable
player_vars = LpVariable.dicts("Player", players, lowBound=0, upBound=1, cat='Integer')

# Objective Function: LpMaximize or LpMinimize
total_score = LpProblem("Fantasy_Points_Problem", LpMaximize) 
 

# go through all the members, looking for the highest point
# Step (2)
total_score += lpSum([project_points[i] * player_vars[i] for i in player_vars])
# Step (3)
total_score += lpSum([salaries[i] * player_vars[i] for i in player_vars]) <= 60000

# Step (4)
pg = [p for p in positions.keys() if positions[p] == 'PG']
sg = [p for p in positions.keys() if positions[p] == 'SG']
sf = [p for p in positions.keys() if positions[p] == 'SF']
pf = [p for p in positions.keys() if positions[p] == 'PF']
c = [p for p in positions.keys() if positions[p] == 'C']

total_score += lpSum([player_vars[i] for i in pg]) == 2
total_score += lpSum([player_vars[i] for i in sg]) == 2
total_score += lpSum([player_vars[i] for i in sf]) == 2
total_score += lpSum([player_vars[i] for i in pf]) == 2
total_score += lpSum([player_vars[i] for i in c]) == 1

total_score.solve()
for v in total_score.variables():
    if v.varValue > 0:
        print(v.name)