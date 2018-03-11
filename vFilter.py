#!/usr/bin/env python
import argparse
import pandas as pd
import ComplementaryFilter
'''
Quaternion math.
conda install -c moble quaternion
'''
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Filtering for IMU.')
    parser.add_argument('--file', dest='file', default='', type=str, help='File containing input data.')
    args = parser.parse_args()
    return args
# 
# Run the filter over a file. 
def filter(file)
    fv = ComplementaryFilter.ComplementaryFilter()
    df = pd.from_csv(file)
    for row in df:
        pass
#
# Main code.
if __name__ == '__main__':
    args = getInputArgs()
    fitler(conf)