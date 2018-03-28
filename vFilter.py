#!/usr/bin/env python
import argparse
import quaternion
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
    parser.add_argument('--isNS', dest='isNS',  action='store_true', help='Divide the TS by 1e9.')
    parser.add_argument('--gt', dest='gt', default=None, type=str, help='File containing input data.')
    args = parser.parse_args()
    return args
# 
# Row to Data.
def getIMUMsg(data, idx, scale):
    row = data.iloc[idx,:]
    msg = ComplementaryFilter.ImuMsg()
    msg.acc = row.as_matrix()[1:4]
    msg.gyro = row.as_matrix()[4:7]
    msg.mag = None
    if len(row) > 8:
        msg.mag = row.as_matrix()[7:10]
    else:
        msg.mag = None
    msg.ts = row.as_matrix()[0] / scale
    return msg
# 
# Run the filter over a file. 
def filter(args):
    df = pd.read_csv(args.file)
    df2 = None
    q_init = None
    if args.gt is not None:
        df2 = pd.read_csv(args.gt)
        # 
        # Handle RPY vs Quat. 
        if df2.as_matrix().shape[1] == 4:
            q_init = quaternion.from_euler_angles(*df2.as_matrix()[0, 1:4])
        else:
            q_init = np.quaternion(*df2.as_matrix()[0, 1:5])
    # 
    # Scale for the ts.
    scale = 1
    if args.isNS:
        scale = 1e9
    # 
    # Get the first message and construct.
    msg = getIMUMsg(df, 0, scale)
    fv = ComplementaryFilter.ComplementaryFilter(msg, q_init)
    # 
    # Collect all states. 
    state = [(msg.ts, fv.getEuler(), fv.getState())]
    # 
    # Drop the first reading and interate
    for rowIdx in range(1,len(df)):
        msg = getIMUMsg(df, rowIdx, scale)
        fv.observation(msg)
        state.append((msg.ts, fv.getEuler(), fv.getState()))
    plot(args, state, df, df2)
    print(fv.gyroBias)

def plot(args, state, df, df2):
    # 
    # PLOTTING
    # 
    ts_arr, state_arr, quat = zip(*state)
    # 
    # Convert to numpy.
    t = np.array(ts_arr)
    s = np.array(state_arr)
    # 
    # GT.
    if args.gt is not None:
        # 
        # Only take check for where there is data.
        maxLen = len(df2)
        maxLen = min(len(df2), len(df))
        q_gt = df2.as_matrix()[:, 1:5]
        ts_gt = df2.as_matrix()[:, 0]
        if args.isNS:
            ts_gt /= 1e9
        # 
        # Check if we got rpy.
        if df2.shape[1] == 4:
            def toQT(row):
                return quaternion.as_float_array(quaternion.from_euler_angles(*row))
            q_gt = np.apply_along_axis(toQT, 1, q_gt)
        quat = np.array(quat)
        ix_start = np.squeeze(np.where(t == ts_gt[0]))
        ix_end = np.squeeze(np.where(t == ts_gt[-1]))
        t = t[ix_start:ix_end]
        quat = quat[ix_start:ix_end+1, :]
        # 
        # Normalize all quaternions. 
        q_gt = q_gt / np.sqrt(np.sum(q_gt ** 2, axis=1, keepdims=True))
        quat = quat / np.sqrt(np.sum(quat ** 2, axis=1, keepdims=True))
        diff = np.arccos(2 * (np.einsum('ij,ij->i', q_gt, quat) ** 2) - 1)
        meanDiff = np.mean(diff)
        print('Mean Orientation Error: ', meanDiff, ' Rads')
        plt.figure()
        plt.title('Error Angle (rad)')
        plt.plot(diff)
        quat_gt = quaternion.as_quat_array(q_gt)
        # 
        # Plot GT RPY.
        rpy = quaternion.as_euler_angles(quat_gt) * 180. / np.pi
        plt.figure()
        plt.title('Ground Truth')
        plt.plot(rpy[:,0], label='Roll')
        plt.plot(rpy[:,1], label='Pitch')
        plt.plot(rpy[:,2], label='Yaw')
        # plt.plot(q_gt[:,0], label='w')
        # plt.plot(q_gt[:,1], label='x')
        # plt.plot(q_gt[:,2], label='y')
        # plt.plot(q_gt[:,3], label='y')
        plt.legend()

    plt.figure()
    plt.title('VF Estimate')
    quat = np.array(quat)
    np.savetxt(args.file + '-result.csv', quat, delimiter=',')
    # plt.plot(quat[:,0], label='w')
    # plt.plot(quat[:,1], label='x')
    # plt.plot(quat[:,2], label='y')
    # plt.plot(quat[:,3], label='y')
    plt.plot(s[:,0], label='Roll')
    plt.plot(s[:,1], label='Pitch')
    plt.plot(s[:,2], label='Yaw')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.show()
#
# Main code.
if __name__ == '__main__':
    args = getInputArgs()
    filter(args)