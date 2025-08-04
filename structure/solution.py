
def createEmptySolution(instance):
    solution = {}
    solution['sol'] = set()
    solution['of'] = 0 # representa el valor de la función objetivo (objective function) asociado con la solución
    solution['instance'] = instance
    return solution


def addToSolution(sol, u, ofVariation = -1):
    if ofVariation == -1:
        for s in sol['sol']:
            sol['of'] += sol['instance']['d'][u][s]
    else:
        sol['of'] += ofVariation
    sol['sol'].add(u)

def distanceToSolution(sol, u, without = -1):
    d = 0
    for s in sol['sol']:
        if s != without:
            print(f"s: {s}\n u: {u}")
            d += sol['instance']['Demand'][s][u+1]
    return round(d, 2)

def isFeasible(sol):

    return len(sol['sol']) == sol['instance']['p']


def printSol(sol):
    print("SOL: "+str(sol['sol']))
    print("OF: "+str(round(sol['of'],2)))