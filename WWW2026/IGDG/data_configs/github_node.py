import os, sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR, "../../data")
processed_datafile_ori = f"{dataroot}/github/github_node"

dataset = "github_node"
testlength = 2
vallength = 2
length = 15