# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:57:40 2025

@author: Vicon-OEM
"""

import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from get_AutoEvents import (get_Events as gev,
                            segments as gs)

main_directory = r'K:\ViconDatabase\Python Code\Py3_Theia_AutoEventDetect\cd3_test_files'
save_folderpath = r'K:\ViconDatabase\Python Code\Py3_Theia_AutoEventDetect\Timing Plots'

def get_c3d_filenamepath():
    # Create a Tkinter root window (hidden)
    root = tk.Tk()
    # root.withdraw()

    # Ask the user to select a folder
    folder_selected = filedialog.askdirectory(initialdir=main_directory, title="Select a Folder")

    if not folder_selected:
        print("No folder selected.")
        return []

    # Get all .c3d file paths in the folder
    c3d_filenamepath = [os.path.join(folder_selected, f) for f in os.listdir(folder_selected) if f.lower().endswith(".c3d")]
    
    root.destroy()
    
    return c3d_filenamepath

# get c3d files list from folder
c3d_filenamepath_list = get_c3d_filenamepath()

# inititialize event dataframe dictionary
ev_dict = {}

# open c3d files and extract events
for filenamepath in c3d_filenamepath_list: 
    print(filenamepath)
    # initialize plot parameters for checking event data
    fig, ax = plt.subplots(2, 1)
    filename = filenamepath.split('\\')[-1]
    fig.suptitle(filename)
    
    # get segment position data and time/frame info into a dataframe using segment labels
    labels = ['foot','toe','pelvis_4X4']
    jerk_threshold = 95
    seg_df, time_df, vfr = gs.get_Segments(filenamepath, save_folderpath, labels, jerk_threshold)
    
    # get relative positions of the foot/toe segments
    reference_segment = 'pelvis_4X4_x'
    rel_df = gs.get_relPosition(seg_df, reference_segment)
    
    # get absolute and relative segment velocities and summed velocities
    mrk_vel, rel_vel, flip_direction = gs.get_segVelocity(seg_df, rel_df, vfr)
    
    #---------------------------------- Foot Off Events -----------------------
    # extract foot off events and show plots
    events_off, samples_perCycle = gev.segment_FootOff(mrk_vel, rel_df, time_df, vfr, ax[0], flip_direction)
    
    #---------------------------------- Foot On Events ------------------------
    # use only x-direction foot segement velocities
    # Theia uses 'Y' as the fore/aft direction
    
    # direction = True
    minThresh = 0.15
    ffq = 0 # filter frequency for foot 'segment' data
    k = 1 # shift
    
    # extract foot off events and show plots
    # print(f'Video Frame Rate is: {vfr}')
    walk_direction = 'x'
    events_onn = gev.segment_FootOn(mrk_vel, rel_df, time_df, vfr, ax[1], minThresh, ffq, k, samples_perCycle, flip_direction, walk_direction)
    
    #save the figure to the filepath
    new_filename = filename.split('.')[0]
    savefilepath = save_folderpath + f'\\On-Off_Events_{new_filename}.png'
    plt.savefig(savefilepath, dpi=300, bbox_inches='tight')
    plt.show()
    
    #--------------------------------- Combine & sort auto events -------------
    events_auto = pd.concat([events_off, events_onn]).sort_values(by='Times', ascending=True)
    
    for index, row in events_auto.iterrows():
        frame = row['Frames']
    # -------------------------------- Pull Manual Events ---------------------
    events_man = gev.c3d_Events(filenamepath, save_folderpath)
    
    # -------------------------------- Trim all df's --------------------------
    if not events_man.empty:
        min_man = events_man['Frames'].min()
        max_man = events_man['Frames'].max()
        events_auto = events_auto[(events_auto['Frames'] > min_man-10) & (events_auto['Frames'] < max_man+10)].copy()
     
    # -------------------------------- Add all dataframes to dictionary -------
    
    ev_dict[f'{filename}'] =  {
        'Auto Events': events_auto,
        'Manual Events': events_man}
    

    
    