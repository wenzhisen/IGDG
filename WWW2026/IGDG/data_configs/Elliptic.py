import os, sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR, "../../data")
processed_datafile_ori = f"{dataroot}/Elliptic/elliptic"
dataset = "elliptic"
testlength = 6
vallength = 6
length = 49