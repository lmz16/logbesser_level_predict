# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:26:39 2019

@author: pei
"""

import os
import random
import shutil
proportion=0.9
pathorigin="corpus/"
pathtrain="data/train/"
pathtest="data/test/"
files= os.listdir(pathorigin)
protofiles=[]
for file in files:
    if("proto" in file):
        protofiles.append(file)
random.shuffle(protofiles)
temp=int(len(protofiles)*proportion)
os.system("rm -f "+pathtrain+"*")
os.system("rm -f "+pathtest+"*")
for i in range(1,temp):
    file1=protofiles[i]
    file2=file1[:-11]+".json"
    shutil.copy(pathorigin+file1,pathtrain)
    shutil.copy(pathorigin+file2,pathtrain)
for i in range(temp,len(protofiles)):
    file1=protofiles[i]
    file2=file1[:-11]+".json"
    shutil.copy(pathorigin+file1,pathtest)
    shutil.copy(pathorigin+file2,pathtest)
