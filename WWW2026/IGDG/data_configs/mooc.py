import os, sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR, "../../data")
processed_datafile_ori = f"{dataroot}/mooc/mooc"

dataset = "mooc"
testlength = 3
vallength = 3
length = 20
