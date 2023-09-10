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
    print(f"{opt.compareA}: {len(resA)}, {opt.compareB}: {len(resB)}")
    print(f"Both {opt.test_case}: {len(intersection)}")
    print(f"{opt.compareA} {opt.test_case}, but {opt.compareB} does not: {len(Bno)}")
    print(f"{opt.compareB} {opt.test_case}, but {opt.compareA} does not: {len(Ano)}")

    with open(logpath, 'a') as f:
        f.write(f"{opt.compareA}: {len(resA)}, {opt.compareB}: {len(resB)}\n")
        f.write(f"Both {opt.test_case}: {len(intersection)}\n")
        f.write(f"{opt.compareA} {opt.test_case}, but {opt.compareB} does not: {len(Bno)}\n")
        f.write(f"{opt.compareB} {opt.test_case}, but {opt.compareA} does not: {len(Ano)}\n")
        f.write('-'*50)
        f.write('\n')
        f.write(f"intersection: {intersection}\n")
        f.write(f"only {opt.compareA} {opt.test_case}: {Bno}\n")
        f.write(f"only {opt.compareB} {opt.test_case}: {Ano}\n")
        f.write('-'*50)
        f.write('\n')
        
    

