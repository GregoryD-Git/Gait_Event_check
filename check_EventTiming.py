# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 12:33:22 2025

@author: dgregory
"""

import Py3_readC3D as pyc3d
import numpy as np
import scipy.signal as signal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# filenamepath = 'K:\\ViconDatabase\\Python Code\\1. readC3D_local\\C3D test files\\TD05MINa03.c3d'
# dir_mk = 'X'

def get_EventDiffs(filenamepath, dir_mk, save_folderpath, x_thresh, data_source):

    def calc_FootOffEventsVel(mk_vel, vfr, ax, markers):
        '''
        Assumption is that foot-off occurs at (about) the peak of the summed marker 
        velocities of the 3 foot markers in the Z-direction for typical heel-toe 
        gait
        '''
        # pull Z-coordinate data first
        mkZ_vel = mk_vel[mk_vel['Coordinate'] == 'Z'].copy()
        
        # Extract trajectory signal
        left_signal = mkZ_vel[markers[0]] + mkZ_vel[markers[1]] + mkZ_vel[markers[2]]
        right_signal = mkZ_vel[markers[3]] + mkZ_vel[markers[4]] + mkZ_vel[markers[5]]
    
        # height must be at least .5 * max
        Lpeaks, _ = signal.find_peaks(left_signal, 
                                     # distance=int(peak_spacing / vfr),
                                     height = 0.5 * max(left_signal))
        
        Rpeaks, _ = signal.find_peaks(right_signal, 
                                     # distance=int(peak_spacing / vfr),
                                     height = 0.5 * max(right_signal))
        
        # Plot the results
        ax.plot(mkZ_vel['Time'], left_signal, label="Trajectory", color='blue')
        ax.plot(mkZ_vel['Time'].iloc[Lpeaks], left_signal.iloc[Lpeaks], "bx")
        ax.plot(mkZ_vel['Time'], right_signal, label="Trajectory", color='red')
        ax.plot(mkZ_vel['Time'].iloc[Rpeaks], right_signal.iloc[Rpeaks], "rx")
        ax.set_xlabel("Time")
        ax.set_ylabel("Trajectory")
        
        # save times and frames to dictionary
        num_Levents = len(Lpeaks)
        num_Revents = len(Rpeaks)
        contexts    = ['Left'] * num_Levents + ['Right'] * num_Revents
        labels      = ['Foot Off'] * (num_Levents + num_Revents)
        times       = list(round(mkZ_vel['Time'].iloc[Lpeaks], 3)) + list(round(mkZ_vel['Time'].iloc[Rpeaks], 3))
        frames      = list(mkZ_vel['Frame'].iloc[Lpeaks]) + list(mkZ_vel['Frame'].iloc[Rpeaks])
        source      = ['Foot Off Markers'] * len(labels)
        
        koff_dict = {
            'Contexts':     contexts,
            'Labels':       labels,
            'Times':        times,
            'Frames':       frames,
            'Source':       source}
        
        events_mk_off = pd.DataFrame(koff_dict).sort_values(by='Times')
        
        return events_mk_off
        
        
    def calc_FootOnEventsVel(mk_vel, vfr, ax, direction, minThresh):
        '''
        Assumption is that the marker on the heel with heel-toe gait will have a 
        near zero velocity in the fowards direction at heel contact. The first frame
        where the heel marker goes below the set threshold of minThresh will be where 
        the first heel contact (i.e. Foot On) event is captured
        '''    
        time    = mk_vel[mk_vel['Coordinate'] == 'X']['Time']
        frame   = mk_vel[mk_vel['Coordinate'] == 'X']['Frame']
        
        # pull heel markers for each foot
        Lheel   = mk_vel[mk_vel['Coordinate'] == 'X'][markers[2]]
        Rheel   = mk_vel[mk_vel['Coordinate'] == 'X'][markers[5]]
        
        # find minThresh% of the max mean absolute X-velocity for each limb
        # foot velocity foot contact threshold - where marker is near zero in x-direction
        Lv_fcT  = minThresh * Lheel.abs().mean().max()
        Rv_fcT  = minThresh * Rheel.abs().mean().max()
        
        # if direction is 'True', velocity peaks are negative, else positive
        if direction:
            # find first index where marker velocity is above threshold for each step
            left_threshold = Lheel > -Lv_fcT
            L_FC = np.where((left_threshold.shift(1, fill_value=False) == False) & left_threshold == True) #.reshape(2,-1)[0].min(axis=1)
            
            right_threshold = Rheel > -Rv_fcT
            R_FC = np.where((right_threshold.shift(1, fill_value=False) == False) & right_threshold == True)
        else:
            # find first index where marker velocity is above threshold for each step
            left_threshold = Lheel < Lv_fcT
            L_FC = np.where((left_threshold.shift(1, fill_value=False) == False) & left_threshold == True) #.reshape(2,-1)[0].min(axis=1)
            
            right_threshold = Rheel < Rv_fcT
            R_FC = np.where((right_threshold.shift(1, fill_value=False) == False) & right_threshold == True)
        
        # save times and frames to dictionary
        num_Levents = len(L_FC[0])
        num_Revents = len(R_FC[0])
        contexts    = ['Left'] * num_Levents + ['Right'] * num_Revents
        labels      = ['Foot Strike'] * (num_Levents + num_Revents)
        times       = list(round(time.iloc[L_FC[0]], 3)) + list(round(time.iloc[R_FC[0]], 3))
        frames      = list(frame.iloc[L_FC]) + list(frame.iloc[R_FC])
        source      = ['Heel Markers'] * len(labels)
        
        kstr_dict = {
            'Contexts':     contexts,
            'Labels':       labels,
            'Times':        times,
            'Frames':       frames,
            'Source':       source}
        
        events_mk_on = pd.DataFrame(kstr_dict).sort_values(by='Times')
        
        return events_mk_on
        
    def calc_ForceEvents(fp_df, threshold):
        fp_cols = [colname for colname in fp_df.columns if colname != 'Time']
        forces_threshold = fp_df[fp_cols] < -threshold
        
        # Find transitions    
        Footstr_indexes = np.array(np.where((forces_threshold.shift(1, fill_value=False) == False) & (forces_threshold == True)))
        FootOff_indexes = np.array(np.where((forces_threshold.shift(1, fill_value=False) == True) & (forces_threshold == False)))
        
        # save times and frames to dictionary
        num_FootOn      = len(Footstr_indexes[0])
        num_FootOff     = len(FootOff_indexes[0])
        Onlabels        = ['Foot Strike'] * num_FootOn
        Offlabels       = ['Foot Off'] * num_FootOff
        labels          = Onlabels + Offlabels
        times           = list(round(fp_df['Time'].iloc[Footstr_indexes[0]], 3)) + list(round(fp_df['Time'].iloc[FootOff_indexes[0]], 3))
        fOn             = fp_df['VideoFrames'].iloc[Footstr_indexes[0]]
        fOff            = fp_df['VideoFrames'].iloc[FootOff_indexes[0]]
        frames          = [int(onframe) for onframe in fOn] + [int(offframe) for offframe in fOff]
        source          = ['Force Plate'] * len(labels)
        
        fc_dict = {
            'Labels':       labels,
            'Times':        times,
            'Frames':       frames,
            'Source':       source}
        
        events_force = pd.DataFrame(fc_dict).sort_values(by='Times')
        
        return events_force
    
    ############################# Extract, Trim, Plot #############################
    
    ###############################################################################
    # --------------------------- Extract C3D data --------------------------------
    # 'getData' is an optional input. If not input, default is all data will be extracted
    C3Ddict = pyc3d.get_C3Ddata(filenamepath)
    subject = C3Ddict['C3D File Name']
    print(subject)
    ###############################################################################
    # --------------------------- Pull force data ---------------------------------
    # Force plate data
    fp_df           = pd.DataFrame()
    fpnums          = list(C3Ddict['Force Data'].keys())
    
    # Pull vertical ground reaction forces for each plate
    for fp_num in fpnums:
        force_plate_num         = fp_num[-1]
        Fz_series               = C3Ddict['Force Data'][f'ForcePlate{force_plate_num}']['Fz']
        fp_df[f'{fp_num}_Fz']   = Fz_series
    
    # analog time
    afr             = 1/C3Ddict['Parameter Group']['ANALOG']['RATE']['data'][0]
    analog_frames   = C3Ddict['Force Data']['ForcePlate1']['AnalogFrames']
    
    # atime
    astart          = afr * analog_frames.iloc[0]
    astop           = afr * analog_frames.iloc[-1]
    atime           = np.linspace(astart, astop, num=len(analog_frames))
    
    # add Time to dataframe
    fp_df['Time'] = atime
    
    
    # melt dataframe for efficient plotting with Seaborn
    fp_cols = [fpname for fpname in fp_df.columns if fpname != 'Time']
    meltFP_df = fp_df.melt(id_vars=['Time'], 
                          value_vars=fp_cols, # column with marker data
                          var_name='Plates', # column with marker names
                          value_name='Force_Z')
    
    # add video frames to dataframe
    fp_df['VideoFrames'] = C3Ddict['Analog Data'].VideoFrames
    ###############################################################################
    # --------------------------- Pull marker data --------------------------------
    mk_df       = C3Ddict['Marker Data']
    if data_source == 'TD':
        markers = ['LMT1H','LANK','LHEE','RMT1H','RANK','RHEE']
    else:
        markers = ['Left_2nd3rd_MT_Head','LANK','Left_Heel','Right_2nd3rd_MT_Head','RANK','Right_Heel']
        
    coordinates = ['X','Z']
    event_ind   = ['On','Off']
    
    # video frame rate
    vfr = 1/C3Ddict['Parameter Group']['POINT']['RATE']['data'][0]
    video_frames = mk_df[mk_df['Coord'] == 'X']['Frame']
    
    # vtime
    vstart = vfr * video_frames.iloc[0]
    vstop = vfr * video_frames.iloc[-1]
    vtime = np.linspace(vstart, vstop, num=len(video_frames))
    
    # use C7 marker to determine walking direction
    C7 = mk_df[mk_df['Coord'] == dir_mk]['C7']
    if C7.iloc[0] > C7.iloc[-1]:
        direction = True # currently unused
    else:
        direction = False
    
    ###############################################################################
    # --------------------------- Set plot parameters -----------------------------
    # Set Seaborn style
    sns.set_style('whitegrid')
    sns.set_palette("coolwarm")  # Use a predefined Seaborn palette
    sns.set_context("paper")  # Adjust font sizes and marker scaling
    fig1, axes1 = plt.subplots(4,1)
    
    # ------------------------- Plot data and event marks -------------------------
    # plot force data
    sns.lineplot(data=meltFP_df, x='Time', y='Force_Z', hue='Plates', ax=axes1[0])
    axes1[0].legend_.remove()
    axes1[0].set_xlabel('')
    axes1[0].set_xticklabels([])
    
    mk_vel = pd.DataFrame()
    
    # plot foot marker data
    for idx, coordinate in enumerate(coordinates):
        axx = idx + 1
        # calculate marker velocity in m/s
        mv = mk_df[mk_df['Coord'] == coordinate][markers].diff().fillna(0) / vfr / 1000
        mv['Time'] = vtime
        mv['Frame'] = list(video_frames)
        mv['Coordinate'] = [coordinate] * len(video_frames)
        
        # melt dataframe
        meltMK_df = mv.melt(id_vars=['Time'], 
                              value_vars=markers, # column with marker data
                              var_name='Marker', # column with marker names
                              value_name='Velocity')
        
        sns.lineplot(data=meltMK_df, x='Time', y='Velocity', hue='Marker', ax=axes1[axx])
        axes1[axx].set_ylabel(f'Foot-{event_ind[axx-1]} \n{coordinate} velocity')
        axes1[axx].set_xlabel('')
        axes1[axx].set_xticklabels([])
        axes1[axx].legend_.remove() 
        
        mk_vel = pd.concat([mk_vel, mv], axis=0)
    
    ###############################################################################
    # ------------------------- Pull events and trim ------------------------------
    minThresh = x_thresh # minimum foot marker velocity threshold is 5% below max average marker velocity
    
    # Gold standard events from file
    gce                 = C3Ddict['Gait Cycle Events']
    events_gold         = gce[['Contexts','Labels','Times','Frames']].copy()
    events_mk_off       = calc_FootOffEventsVel(mk_vel, vfr, axes1[3], markers)
    events_mk_on        = calc_FootOnEventsVel(mk_vel, vfr, axes1[1], direction, minThresh)
    events_force        = calc_ForceEvents(fp_df, 10) # dataframe and force threshold
    
    # add source info for Gait Cycle Events dataframe
    source_force        = ['Gold Standard'] * len(events_gold)
    events_gold.loc[:,'Source'] = source_force
    
    # all dataframes need to have events calculated from the same true event so need to be trimmed
    # find trimming thresholds by 'Frames' 
    ev_cutoff = 10
    minFO_events        = max(min(events_force[events_force['Labels'] == 'Foot Off']['Frames'] - ev_cutoff), min(events_gold[events_gold['Labels'] == 'Foot Off']['Frames'] - ev_cutoff))
    minFS_events        = max(min(events_force[events_force['Labels'] == 'Foot Strike']['Frames'] - ev_cutoff), min(events_gold[events_gold['Labels'] == 'Foot Strike']['Frames'] - ev_cutoff))
    maxFO_events        = min(max(events_force[events_force['Labels'] == 'Foot Off']['Frames'] + ev_cutoff), max(events_gold[events_gold['Labels'] == 'Foot Off']['Frames'] + ev_cutoff))
    maxFS_events        = min(max(events_force[events_force['Labels'] == 'Foot Strike']['Frames'] + ev_cutoff), max(events_gold[events_gold['Labels'] == 'Foot Strike']['Frames'] + ev_cutoff))
    
    # trim dataframes with above thresholds
    events_goldOffT     = events_gold[(events_gold['Frames'] > minFO_events) & (events_gold['Frames'] < maxFO_events) & (events_gold['Labels'] == 'Foot Off')].copy() # events_gtrim[events_gtrim['Labels'] == 'Foot Off']
    events_goldStrT     = events_gold[(events_gold['Frames'] > minFS_events) & (events_gold['Frames'] < maxFS_events) & (events_gold['Labels'] == 'Foot Strike')].copy() # events_gtrim[events_gtrim['Labels'] == 'Foot Strike']
    events_forceOffT    = events_force[(events_force['Frames'] > minFO_events) & (events_force['Frames'] < maxFO_events) & (events_force['Labels'] == 'Foot Off')].copy() # events_ftrim[events_ftrim['Labels'] == 'Foot Off']
    events_forceStrT    = events_force[(events_force['Frames'] > minFS_events) & (events_force['Frames'] < maxFS_events) & (events_force['Labels'] == 'Foot Strike')].copy() # events_ftrim[events_ftrim['Labels'] == 'Foot Strike']
    events_markrOffT    = events_mk_off[(events_mk_off['Frames'] > minFO_events) & (events_mk_off['Frames'] < maxFO_events) & (events_mk_off['Labels'] == 'Foot Off')].copy() # events_mk_off[(events_mk_off['Frames'] > min_events) & (events_mk_off['Frames'] < max_events)]
    events_markrStrT    = events_mk_on[(events_mk_on['Frames'] > minFS_events) & (events_mk_on['Frames'] < maxFS_events) & (events_mk_on['Labels'] == 'Foot Strike')].copy() # events_mk_on[(events_mk_on['Frames'] > min_events) & (events_mk_on['Frames'] < max_events)]
    
    # need to add indicators for each event in each dataframe so unique events can be compared later
    # need to do this once for foot off events and once for foot strike events
    off_ID      = 0
    off_list    = []
    for idx, _ in events_goldOffT.iterrows():
        off_string = f'event_{off_ID}'
        off_ID += 1
        off_list.append(off_string)
        
    str_ID       = 0
    str_list     = []
    for idx, _ in events_goldStrT.iterrows():
        str_string = f'event_{str_ID}'
        str_ID += 1
        str_list.append(str_string)
        
    # add event ID lists to dataframes
    events_goldOffT['EventID']  = off_list
    events_forceOffT['EventID'] = off_list
    events_markrOffT['EventID'] = off_list
    events_goldStrT['EventID']  = str_list
    events_forceStrT['EventID'] = str_list
    events_markrStrT['EventID'] = str_list
    
    # ------------------------- Add lines for events ------------------------------
    fmin = min(meltFP_df['Force_Z'])
    fmax = max(meltFP_df['Force_Z'])
    # line_length = abs(fmax - fmin)
    
    # Plot vertical lines for auto gait events
    for item, row in events_forceOffT.iterrows():
        color = 'black'
        axes1[0].vlines(row['Times'], ymin=fmin, ymax=fmax, color=color, linewidth=0.5, linestyle='-')
        
    for _, row in events_forceStrT.iterrows():
        color = 'black'
        axes1[0].vlines(row['Times'], ymin=fmin, ymax=fmax, color=color, linewidth=0.5, linestyle='--')
    
    for _, row in events_goldOffT.iterrows():
        color = 'blue' if row['Contexts'] == 'Left' else 'red'
        axes1[0].vlines(row['Times'], ymin=fmin, ymax=fmax, color=color, linewidth=0.5, linestyle='-')
        
    for _, row in events_goldStrT.iterrows():
        color = 'blue' if row['Contexts'] == 'Left' else 'red'
        axes1[0].vlines(row['Times'], ymin=fmin, ymax=fmax, color=color, linewidth=0.5, linestyle='--')
    
    for _, row in events_markrOffT.iterrows():
        color = 'green' if row['Contexts'] == 'Left' else 'magenta'
        axes1[0].vlines(row['Times'], ymin=fmin, ymax=fmax, color=color, linewidth=0.5, linestyle='-')
        
    for _, row in events_markrStrT.iterrows():
        color = 'green' if row['Contexts'] == 'Left' else 'magenta'
        axes1[0].vlines(row['Times'], ymin=fmin, ymax=fmax, color=color, linewidth=0.5, linestyle='--')
    
    # ------------------------- Export figure -------------------------------------
    # figure properties
    plt.suptitle('Foot marker velocity v. Force threshold', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Replace 'output.pdf' with your desired PDF filename
    # subject = C3Ddict['C3D File Name']
    png_filename = f'{subject}_Trajectory_Visualization.png'
    save_png_path = os.path.join(save_folderpath, png_filename)
    fig1.savefig(save_png_path, dpi=300, bbox_inches='tight')
    
    ###############################################################################
    # ------------------------- Plot only forces and events -----------------------
    fig2, axes2 = plt.subplots(1,1)
    
    # plot force data
    sns.lineplot(data=meltFP_df, x='Time', y='Force_Z', hue='Plates', ax=axes2)
    axes2.legend_.remove()
    axes2.set_xlabel('')
    axes2.set_xticklabels([])
    axes2.set_xlim(events_force['Times'].min() - 0.15, events_force['Times'].max() + 0.15)
    
    # Plot vertical lines for auto gait events
    for _, row in events_forceOffT.iterrows():
        color = 'black'
        axes2.vlines(row['Times'], ymin=fmin, ymax=fmax, color=color, linewidth=0.5, linestyle='-')
        
    for _, row in events_forceStrT.iterrows():
        color = 'black'
        axes2.vlines(row['Times'], ymin=fmin, ymax=fmax, color=color, linewidth=0.5, linestyle='--')
    
    for _, row in events_goldOffT.iterrows():
        color = 'blue' if row['Contexts'] == 'Left' else 'red'
        axes2.vlines(row['Times'], ymin=fmin, ymax=fmax, color=color, linewidth=0.5, linestyle='-')
        
    for _, row in events_goldStrT.iterrows():
        color = 'blue' if row['Contexts'] == 'Left' else 'red'
        axes2.vlines(row['Times'], ymin=fmin, ymax=fmax, color=color, linewidth=0.5, linestyle='--')
    
    for _, row in events_markrOffT.iterrows():
        color = 'purple' if row['Contexts'] == 'Left' else 'orange'
        axes2.vlines(row['Times'], ymin=fmin, ymax=fmax, color=color, linewidth=0.5, linestyle='-')
        
    for _, row in events_markrStrT.iterrows():
        color = 'green' if row['Contexts'] == 'Left' else 'magenta'
        axes2.vlines(row['Times'], ymin=fmin, ymax=fmax, color=color, linewidth=0.5, linestyle='--')
        
    # figure properties
    plt.suptitle('Forces with events', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Replace 'output.pdf' with your desired PDF filename
    png_filename = f'{subject}_Event_Comparison.png'
    save_png_path = os.path.join(save_folderpath, png_filename)
    fig2.savefig(save_png_path, dpi=300, bbox_inches='tight')
    
    ###############################################################################
    # --------------------------- Combine and calc event diff ---------------------
    # gold standard events versus forces events
    gfOff_df        = pd.concat([events_goldOffT, events_forceOffT], axis=0).sort_values('Frames')
    gfOff_diff      = gfOff_df.groupby('EventID')['Frames'].diff().dropna()
    gfStr_df        = pd.concat([events_goldStrT, events_forceStrT], axis=0).sort_values('Frames')
    gfStr_diff      = gfStr_df.groupby('EventID')['Frames'].diff().dropna()
    
    fmOff_df        = pd.concat([events_forceOffT, events_markrOffT], axis=0).sort_values('Frames')
    fmOff_diff      = fmOff_df.groupby('EventID')['Frames'].diff().dropna()
    fmStr_df        = pd.concat([events_forceStrT, events_markrStrT], axis=0).sort_values('Frames')
    fmStr_diff      = fmStr_df.groupby('EventID')['Frames'].diff().dropna()
    
    gmOff_df        = pd.concat([events_goldOffT, events_markrOffT], axis=0).sort_values('Frames')
    gmOff_diff      = gmOff_df.groupby('EventID')['Frames'].diff().dropna()
    gmStr_df        = pd.concat([events_goldStrT, events_markrStrT], axis=0).sort_values('Frames')
    gmStr_diff      = gmStr_df.groupby('EventID')['Frames'].diff().dropna()
    
    FootOff_dict = {
        'Gold_Force':   list(gfOff_diff),
        'Force_Marker': list(fmOff_diff),
        'Gold_Marker':  list(gmOff_diff),
        'Subject':      [subject] * len(gfOff_diff)
        }
    
    FootStr_dict = {
        'Gold_Force':   list(gfStr_diff),
        'Force_Marker': list(fmStr_diff),
        'Gold_Marker':  list(gmStr_diff),
        'Subject':      [subject] * len(gfStr_diff)
        }
    
    FootOff_df = pd.DataFrame(FootOff_dict)
    FootStr_df = pd.DataFrame(FootStr_dict)
    
    return FootOff_df, FootStr_df
