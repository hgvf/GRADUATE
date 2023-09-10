import os
import glob
import numpy as np
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--compareA', type=str)
    parser.add_argument('--compareB', type=str)
    parser.add_argument('--test_case', type=str)

    opt = parser.parse_args()

    return opt

def collect_idx(opt, files):
    res = []
    for f in files:
        tmp = f.split('_')
        if tmp[0] != opt.test_case:
            continue

        res.append(int(tmp[1][:-4]))
    return res

if __name__ == '__main__':
    opt = parse_args()

    path_A = f"./plot/{opt.compareA}"
    path_B = f"./plot/{opt.compareB}"

    logpath = f"./log/compare_{opt.compareA}_{opt.compareB}.log"

    filesA = os.listdir(f"{path_A}")
    filesB = os.listdir(f"{path_B}")

    resA = set(collect_idx(opt, filesA))
    resB = set(collect_idx(opt, filesB))

    intersection = resA.intersection(resB)
    Bno = resA - resB
    Ano = resB - resA
    print(f"resA: {len(resA)}, resB: {len(resB)}")
    print(f"Both {opt.test_case}: {len(intersection)}")
    print(f"A {opt.test_case}, but B does not: {len(Bno)}")
    print(f"B {opt.test_case}, but A does not: {len(Ano)}")

    with open(logpath, 'a') as f:
        f.write(f"resA: {len(resA)}, resB: {len(resB)}\n")
        f.write(f"Both {opt.test_case}: {len(intersection)}\n")
        f.write(f"A {opt.test_case}, but B does not: {len(Bno)}\n")
        f.write(f"B {opt.test_case}, but A does not: {len(Ano)}\n")
        f.write('-'*50)
        f.write('\n')
        f.write(f"intersection: {intersection}\n")
        f.write(f"only A {opt.test_case}: {Bno}\n")
        f.write(f"only B {opt.test_case}: {Ano}\n")
        f.write('-'*50)
        f.write('\n')
        
    

