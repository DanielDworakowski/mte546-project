#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
import ComplementaryFilter
import matplotlib.pyplot as plt

'''
Quaternion math.
conda install -c moble quaternion
'''
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Filtering for IMU.')
    parser.add_argument('--file', dest='file', default='example.csv', type=str, help='File containing input data.')
    args = parser.parse_args()
    return args
# 
# Row to Data.
def getIMUMsg(data, idx):
    row = data.iloc[idx,:]
    msg = ComplementaryFilter.ImuMsg()
    msg.acc = row.as_matrix()[1:4]
    msg.gyro = row.as_matrix()[4:7]
    msg.mag = row.as_matrix()[7:10]
    msg.ts = row.as_matrix()[0]
    return msg
# 
# Run the filter over a file. 
def filter(args):
    df = pd.read_csv(args.file)
    # 
    # Get the first message and construct.
    msg = getIMUMsg(df, 0)
    fv = ComplementaryFilter.ComplementaryFilter(msg)
    # 
    # Collect all states. 
    state = [(msg.ts, fv.getEuler())]
    # 
    # Drop the first reading and interate
    for rowIdx in range(1,len(df)):
        msg = getIMUMsg(df, rowIdx)
        fv.observation(msg)
        state.append((msg.ts, fv.getEuler()))
    ts_arr, state_arr = zip(*state)
    # 
    # Convert to numpy.
    t = np.array(ts_arr)
    s = np.array(state_arr)
    plt.plot(t, s[:,0], 'r', label='Roll')
    plt.plot(t, s[:,1], 'b', label='Pitch')
    plt.plot(t, s[:,2], 'g', label='Yaw')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.show()
#
# Main code.
if __name__ == '__main__':
    args = getInputArgs()
    filter(args)