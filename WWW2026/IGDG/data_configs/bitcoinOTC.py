import os, sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR, "../../data")
processed_datafile_ori = f"{dataroot}/original/bitcoinOTC"
processed_datafile_eva = f"{dataroot}/evasive/bitcoinOTC"
processed_datafile_poi = f"{dataroot}/poisoning/bitcoinOTC"

dataset = "bitcoinOTC"
testlength = 2
vallength = 2
length = 6