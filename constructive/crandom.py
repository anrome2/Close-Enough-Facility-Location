from structure import solution, instances
import random

def construct(inst, p, t, R):
    sol = solution.createEmptySolution(inst)
    n = inst['Customer'].max()
    # como en este caso las facilities serán localizaciones potenciales de instalaciones, y el número de instalaciones es p, elegiremos de forma aleatoria p localizaciones
    facilities = []
    for i in range(p):
        u = random.randint(0,n-1) # falta comprobar que no se repita la instalación
        solution.addToSolution(sol, u)
        facilities.append(u)
    cl = createCandidateList(sol, u)
    # alpha representa la probabilidad de escoger un cliente aleatorio
    alpha = random.random()
    while not solution.isFeasible(sol):
        gmin, gmax = evalGminGmax(cl)
        th = gmax - alpha * (gmax - gmin)
        rcl = []
        for i in range(len(cl)):
            if cl[i][0] >= th:
                rcl.append(cl[i])
        selIdx = random.randint(0, len(rcl)-1)
        cSel = rcl[selIdx]
        solution.addToSolution(sol, cSel[1], cSel[0])
        cl.remove(cSel)
        updateCandidateList(sol, cl, cSel[1])
    return sol


def createCandidateList(sol, first):
    n = sol['instance']['Customer'].max()
    cl = []
    for c in range(n):
        if c != first:
            d = solution.distanceToSolution(sol, c)
            cl.append([d,c])
    return cl

def evalGminGmax(cl):
    gmin = 0x3f3f3f3f # Num. muy grande
    gmax = 0
    for c in cl:
        gmin = min(gmin, c[0])
        gmax = max(gmax, c[0])
    return gmin, gmax


def updateCandidateList(sol, cl, added):
    for i in range(len(cl)):
        c = cl[i]
        c[0] += sol['instance']['Demand'][added][c[1]]