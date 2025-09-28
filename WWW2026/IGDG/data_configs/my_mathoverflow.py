import os, sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR, "../../data")
processed_datafile_ori = f"{dataroot}/original/mathoverflow"
processed_datafile_eva = f"{dataroot}/evasive/mathoverflow"
processed_datafile_poi = f"{dataroot}/poisoning/mathoverflow"

dataset = "mathoverflow"
testlength = 5
vallength = 5
length = 25