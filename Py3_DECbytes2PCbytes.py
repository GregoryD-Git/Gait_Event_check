# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:20:30 2024

@author: daniel.gregory@shrinenet.org
"""

# DEC to bytes function
import numpy as np

def DEC2PC(dec_bytes):
    # try: 
    pc_bytes = bytes([dec_bytes[2], dec_bytes[3], dec_bytes[0], dec_bytes[1]])
    # except:
        # print('stop')
    
    my_bytes = np.frombuffer(pc_bytes, dtype=np.float32, count=1)[0]/4
    return my_bytes

if __name__ == "__main__":
    # test1.py executed as a script
    DEC2PC()