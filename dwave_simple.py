import sys
import numpy as np

import dimod
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

def loadFile(filename):
    commands= {}
    with open(filename) as fh:
        for line in fh:
            if line[0] != "#":
                command, description = line.strip().split('\t', 1)
                commands[command] = description.strip()
    return commands

def coef2Prob(commands):
    linear=dict()
    quadratic=dict()
    N=int(commands["N"])
    for i in range(0,N):
        linear[i]=float(commands["h{}".format(i)])
    for i in range(1,N):
        for j in range(0,i):
            quadratic[(i,j)]=float(commands["J{}_{}".format(i,j)])
    const=float(commands["const"])
    return quadratic, linear, const

if __name__ == '__main__':
    quadratic, linear, const=coef2Prob(loadFile(sys.argv[1]))
    bqm=dimod.BinaryQuadraticModel(linear,quadratic,const,dimod.Vartype.SPIN)
    solver=EmbeddingComposite(DWaveSampler())

    computation=solver.sample(bqm, num_reads=1000, annealing_time=20)
    print("QPU time used:", computation.info['timing']['qpu_access_time'], "microseconds.")

    print("#histogram")
    hist=dict()
    for i in range(len(computation.record)):
        ene=computation.record[i]['energy']
        cnt=computation.record[i]['num_occurrences']
        if ene in hist:
            hist[ene]+=cnt
        else:
            hist[ene]=cnt
    for x in hist:
        print(x,"\t",hist[x])


    energy_vec=computation.data_vectors['energy']
    i_best=np.argmin(energy_vec)
    energy=computation.record[i_best]['energy']
    print("energy=",energy)
    vec=computation.record[i_best]['sample']
    print("vec=",vec)
    np.save("solution_vec",vec)
