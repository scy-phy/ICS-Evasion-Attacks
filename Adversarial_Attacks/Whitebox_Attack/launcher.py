import os
import subprocess
import sys
dataset = 'BATADAL'
if dataset == 'BATADAL':
    intervals = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14]]
    plcs = ['PLC_1','PLC_2', 'PLC_3', 'PLC_4', 
            'PLC_5', 'PLC_6', 'PLC_7', 'PLC_8', 'PLC_9'
            ]
if dataset == 'WADI':
    intervals = [[1,2],[3,5],[6,7],[8,9],[10,11],[12,13],[14,15]]
    plcs =  ['PLC_1','PLC_2']
for interval in intervals:
    print('python constrained_attack_PLC.py -a '+ dataset +' -b '+ str(interval[0])+' '+ str(interval[1])+' -c'+str(plcs))
    pid = subprocess.Popen(['python', 'constrained_attack_PLC.py', '-a', dataset,'-b', str(interval[0]), str(interval[1]), '-c'] + plcs ) # Call subprocess
    #os.system() 
