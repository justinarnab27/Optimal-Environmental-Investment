import numpy as np

from copy import copy, deepcopy
from itertools import combinations


def PL(k, j, s):
    '''Vacancy loss per unit excess pollution treatment capacity'''
    return LO(k, j, s)/(s[k][j] - p[k][j])


def TI(j):
    '''Present Value of total investment of jth project'''
    sum_var = 0
    for i in range(1, N[j]+1):
        sum_var += I[i-1][j] * ((1 + r)**(N[j] - i + 1))
        return sum_var


def CP(j):
    '''Investment per unit increase in pollution treatment capacity'''
    return TI(j)/PT[j]


def PLp(k, j, s):
    '''Loss in multiple year of vacancy'''
    sum_var = 0
    #Sum PL for last years when there was excess pollution treatment capacity
    for i in range(1, N[j]+1):
        if s[k-i][j] > p[k-i][j]:
            sum_var += PL(k-i, j, s)
        else:
            break
    return sum_var


def LO(k, j, s):
    '''Vacancy loss'''
    if s[k][j] > p[k][j]:
        return r * (s[k][j] - p[k][j]) * (CP(j) + PLp(k, j, s))
    else:
        return 0


def PN(k, j, s):
    '''Penalty Loss'''
    if p[k][j] > s[k][j]:
        return (p[k][j] - s[k][j])*pf[k]
    else:
        return 0


def D(k, s):
    '''Contribution function'''
    s = deepcopy(s)
    sum_var = 0
    los = 0
    pns = 0
    #Sum the losses due to penalty or vacancy
    for j in range(M):
        lov = LO(k, j, s)
        los += lov
        pn = PN(k, j, s)
        pns += pn
        sum_var += lov*100 + pn
    return sum_var


def find_feasible_proj(k, proj_finished_now, curr_investment):
    '''Finds the project combinations that sasisfy the constraints'''
    curr_investment = deepcopy(curr_investment)
    all_combinations = []
    all_feasible = []
    #Finds all combinations of projects that can finish on kth year
    for i in range(len(proj_finished_now)+1):
        for j in combinations(proj_finished_now, i):
            all_combinations.append(j)
    #Checks if the project combination does not exceed MaxI investment per year
    for comb in all_combinations:
        is_feasible = True
        #Calculates all the investment required for the project combination
        for j in comb:
            for i in range(N[j]):
                curr_investment[k - N[j] + i][j] += I[i][j]
        #Checks if Maximum investment limit isn't exceeded
        for i in range(T):
            sum_var = 0
            for j in range(M):
                sum_var += curr_investment[i][j]
            if sum_var > MaxI:
                is_feasible = False
        #If all constraints are satisfied then adds to the is_feasible list
        if is_feasible:
            all_feasible.append(comb)
        #Reverses the changes done to curr_investment
        for j in comb:
            for i in range(N[j]):
                curr_investment[k - N[j] + i][j] -= I[i][j]

    return all_feasible


def stage_k(k, s, available_project, f_last, curr_investment):
    '''Simulates stage k of the model'''
    global final_f      
    global final_investment
    curr_investment = deepcopy(curr_investment)
    #If the final stage is reached then finds out the best investment 
    if k == T:
        if f_last < final_f:
            final_f = f_last
            final_investment = curr_investment
        return
    proj_finished_now = []
    #Finds out which projects have not already been done
    for i in available_project:
        if N[i] < k+1:
            proj_finished_now.append(i)
    #Calculate all the project combinations that satisfy the constraint
    all_feasible = find_feasible_proj(k, proj_finished_now, curr_investment)

    #Loop over all feasible combinations and calculate D and f
    for comb in all_feasible:
        now_available = deepcopy(available_project)
        now_investment = deepcopy(curr_investment)
        now_s = deepcopy(s)
        X = np.array([0, 0, 0, 0, 0])
        for j in comb:
            X[j] = PT[j]
            now_available.remove(j)
            for i in range(N[j]):
                now_investment[k - N[j] + i][j] += I[i][j]
        if k > 0:
            now_s[k, :] = now_s[k-1, :] + X
        D_val = D(k, now_s)     #Caulculate contribution function
        f_val = f_last + D_val   #Calculate optimal value function
        #Go to the next stage
        stage_k(k+1, now_s, now_available, f_val, now_investment)


def solve(S):
    '''Find the optimal investment plan'''
    S = np.array(S)
    available_project = set(range(M))
    s = np.zeros(shape=(T, M))
    s[0, :] = S
    curr_investment = np.zeros(shape=(T, M))
    stage_k(0, s, available_project, 0, curr_investment)

final_f = 999999             #Minimum loss
final_investment = None       #Optimal Investment Plan

M = 5  # number of projects
T = 6  # The number of periods
r = 0.06  # Discount rate
# Pollution treatment capacity after completion of project j
PT = np.array([70, 270, 110, 70, 200])
N = [2, 4, 3, 2, 4]  # Number of years necessary to finish project j
# Investment at year i and project j is I[i][j]
I = np.array([[3.5, 5.1, 4, 2.5, 7.1],
              [4, 3.2, 3.7, 2.5, 6.5],
              [0, 1.8, 4.8, 0, 3.2],
              [0, 1.8, 0, 0, 4]])
# I = 1e6 * I
# Maximum pollution level in year i of type j is p[i][j]
p = np.array([[47, 100, 100, 30, 140],
              [63, 140, 120, 41, 175],
              [76, 180, 140, 50, 210],
              [85, 220, 155, 63, 245],
              [90, 290, 165, 75, 270],
              [105, 350, 178, 88, 290]])
# The state. The pollution capacity at year i of type j is s[i][j]
s = np.zeros(shape=(6, 5))
pf = [1, 1.5, 3, 4.7, 5.5, 6.5]  # Penalty for pollution
MaxI = 13  # maximum investment can be made in a yer

solve([40, 80, 70, 20, 100])

print(f"The Minimum Loss is: {round(final_f/100,2)} Million Yuan")
print("Optimal Investment Plan is: ")
print(final_investment)
