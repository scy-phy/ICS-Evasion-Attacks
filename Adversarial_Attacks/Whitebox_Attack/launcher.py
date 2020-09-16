import os
import subprocess
import sys
"""
    This script speeds up the computation of the topology based attack. 
    Specifically it creates multiple subprocesses and assigns to it a subset of the attack data to conceal.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', nargs='+', type=str, default=['BATADAL'])
args = parser.parse_args()
print(args.data)

dataset = args.data[0]

if dataset == 'BATADAL':
    intervals = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14]]
    plcs = ['PLC_1','PLC_2', 'PLC_3', 'PLC_4', 
            'PLC_5', 'PLC_6', 'PLC_7', 'PLC_8', 'PLC_9'
            ]
if dataset == 'WADI':
    intervals = [[1,2],[3,5],[6,7],[8,9],[10,11],[12,13],[14,15]]
    plcs =  ['PLC_1','PLC_2']
for interval in intervals:
    print('python constrained_attack_PLC.py -d '+ dataset +' -a '+ str(interval[0])+' '+ str(interval[1])+' -p'+str(plcs))
    pid = subprocess.Popen(['python', 'constrained_attack_PLC.py', '-d', dataset,'-a', str(interval[0]), str(interval[1]), '-p'] + plcs ) # Call subprocess
    #os.system() 
