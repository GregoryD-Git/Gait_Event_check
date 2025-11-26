# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 12:33:22 2025

@author: dgregory
"""

from ezc3d import c3d as ez
import numpy as np
import scipy.signal as signal
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# from visual_Skelton import animate_Theia as animate
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation

# %matplotlib qt
class signal_process():
    def segment_Filter(raw_signal, ffq, vfr):
        # Design the Butterworth band-pass filter
        low_cut = ffq # vfr/2 - 1  # Low cutoff frequency (Hz)
        order = 4  # Filter order
        
        # Get filter coefficients using 'fs' to specify frequency in Hz
        b, a = signal.butter(order, low_cut, btype='low', fs=vfr)
        
        # Apply the filter using filtfilt (zero-phase filtering)
        filt_signal = signal.filtfilt(b, a, raw_signal)
        
        return filt_signal
    
    def dominant_Frequency(trajectory, vfr):
        N = len(trajectory)
        # fft
        fft_values = fft(trajectory.values)
        frequencies = fftfreq(N, 1/vfr)
        # dominant frequency
        idx = np.argmax(np.abs(fft_values[:N//2]))
        dom_freq = frequencies[idx]
        
        return dom_freq
    
    def compute_jerk(df, segment_names):
        # Computing the norm magnitude for each segment
        # Compute velocity
        velocity = df[segment_names].diff()
        # Compute acceleration
        acceleration = velocity.diff()
        # Compute jerk
        jerk = acceleration.diff()
        # Jerk magnitude
        jerk_mag = np.linalg.norm(jerk.values, axis=1)
        return jerk_mag
    
    def compute_angles(df, segment_a_cols, segment_b_cols):
        # Vector from A to B
        vec = df[segment_b_cols].values - df[segment_a_cols].values
        # Normalize
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        vec_normalized = vec / np.where(norm == 0, 1, norm)
        # Angle between consecutive vectors
        dot = np.sum(vec_normalized[:-1] * vec_normalized[1:], axis=1)
        angles = np.arccos(np.clip(dot, -1.0, 1.0))  # in radians
        return angles
    
    def compute_spectral_entropy(trajectory, fs, window_size=128, step_size=64):
        entropies = []
        for start in range(0, len(trajectory) - window_size, step_size):
            segment = trajectory[start:start + window_size]
            freqs, psd = signal.welch(segment, fs=fs)
            psd_norm = psd / np.sum(psd)
            ent = entropy(psd_norm)
            entropies.append(ent)
        return np.array(entropies)

class get_Events():
    def segment_FootOff(mrk_vel, rel_df, time_df, vfr, ax, flip_direction):
        # initialize the events dataframe
        events_df = pd.DataFrame()
        
        # time and frame data
        time = time_df['Time']
        frame = time_df['Frame']
        
        # obtain summed signal_vel dominant frequency to specify peak spacing parameter for each column
        segment_names = [name for name in mrk_vel.columns if 'sum' in name]
        for segment in segment_names:
            trajectory = mrk_vel[segment]
            dom_freq = signal_process.dominant_Frequency(trajectory, vfr)
            # windowing to within half the number of samples per cycle
            samples_perCycle = int(vfr / dom_freq / 2)
            
            # height must be at least .5 * max
            s_peaks, _ = signal.find_peaks(trajectory, 
                                         height = 0.5 * max(trajectory),
                                         )
            
            # only pull largest peaks within window of anticipated steps 
            # this bit of code takes only the peaks within +/- samples_perCycle of the peak
            # - this will reject smaller peaks identified that are not equal to the max value in that range 
            signal_peaks = [peak for peak in s_peaks if trajectory[peak] == max(trajectory[max(0, peak-samples_perCycle):min(len(trajectory), peak+samples_perCycle)])]
        
            # -------------- Get relative position to filter events -----------
            # pull relative segment positions to indentify when the foot is behind the body
            if 'l_foot' in segment:
                relative_segment = 'l_foot_4X4_x'
            else:
                relative_segment = 'r_foot_4X4_x'
                
            if flip_direction:
                # scaling segment data just for plotting purposes
                rel_segment = -1 * rel_df[relative_segment]/1000
            else:
                rel_segment = rel_df[relative_segment]/1000
            
            # make bool series to filter events - True when relative position is negative (i.e. behind body)
            x_behind = rel_segment < 0
            
            # ------------ Filter events based on relative position -----------
            event_indexes = [event for event in signal_peaks if x_behind[event]]
            
            # specify sides
            if 'l_' in segment:
                side = 'Left'
                plot_color = 'blue'
                ax.plot(time.iloc[event_indexes], trajectory[event_indexes], "kx")
                # ax.plot(time.iloc[event_indexes], rel_df['l_foot_4X4_x'][event_indexes]/1000, 'kx')
            else:
                side = 'Right'
                plot_color = 'red'
                ax.plot(time.iloc[event_indexes], trajectory[event_indexes], "kx")
                # ax.plot(time.iloc[event_indexes], rel_df['r_foot_4X4_x'][event_indexes]/1000, 'kx')
    
            # Plot the results
            ax.plot(time, trajectory, label=segment, color=plot_color)
            ax.plot(time, rel_segment, label='relative position', color=plot_color, linewidth=0.5, linestyle='--')
            ax.set_xlabel("Time")
            ax.set_ylabel("Foot Off Events\nVertical Velocity")
            # sets the upper left anchor point of the legend to 100% above and right of axis origin
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            # save times and frames to dictionary
            num_events = len(event_indexes)
            
            # assing values to dictionary
            contexts    = [side] * num_events
            labels      = ['Foot Off'] * num_events
            times       = list(round(time.iloc[event_indexes], 3))
            frames      = list(frame.iloc[event_indexes])
            source      = ['Foot & Toe Segments'] * len(labels)
            
            koff_dict = {
                'Contexts':     contexts,
                'Labels':       labels,
                'Times':        times,
                'Frames':       frames,
                'Source':       source}
            
            df = pd.DataFrame(koff_dict)
        
            events_df = pd.concat([events_df, df])
        
        # sort all events
        events_df.sort_values(by='Frames') # ascending order
    
        return events_df, samples_perCycle
        
    def segment_FootOn(mrk_vel, rel_df, time_df, vfr, ax, minThresh, ffq, k, samples_perCycle, flip_direction, walk_direction):
        # initialize events dataframe
        events_df = pd.DataFrame()
        
        # time and frame data
        time        = time_df['Time']
        frame       = time_df['Frame']
            
        # loop through and get foot segment data
        segment_names = [name for name in mrk_vel.columns if 'foot' in name and walk_direction in name]
        for segment in segment_names:
            if ffq == 0:
                signal_vel = mrk_vel[segment]
            else:
                raw_signal = mrk_vel[segment]
                signal_vel = pd.Series(signal_process.segment_Filter(raw_signal, ffq, vfr)) 
            
            # if flip_direction is 'True', velocity peaks are negative so flip, else positive
            if flip_direction:
                # flip signal so peaks are positive
                signal_vel = -signal_vel
                
            # find minThresh% of the max mean absolute X-velocity for each limb
            # foot velocity foot contact threshold - where segment is near zero in x-flip_direction    
            v_fcT  = minThresh * signal_vel.max()
            
            # ---------------------------------------------------------------------
            # find first index where segment velocity is below threshold for each step
            threshold = signal_vel < v_fcT
            
            # Identify the rising edge (False → True)
            signal_FC = threshold & ~threshold.shift(k, fill_value=True)
            
            # Get the index positions
            fc_indexes = signal_FC[signal_FC].index
            
            # -------------- Get relative position to filter events -----------
            # pull relative segment positions to indentify when the foot is behind the body
            if 'l_foot' in segment:
                relative_segment = 'l_foot_4X4_x'
            else:
                relative_segment = 'r_foot_4X4_x'
                
            if flip_direction:
                # scaling segment data just for plotting purposes
                rel_segment = -1 * rel_df[relative_segment]/1000
            else:
                rel_segment = rel_df[relative_segment]/1000
            
            # make bool series to filter events - True when relative position is positve (i.e. ahead of body)
            x_behind = rel_segment > 0
            
            # ------------ Filter events based on relative position -----------
            unfiltEvent_indexes = [event for event in fc_indexes if x_behind[event]]
            
            # ------------ Filter out events less than expected sampling -----------
            reject_events = np.where(np.diff(np.array(unfiltEvent_indexes)) < samples_perCycle)[0] # + 1
            event_indexes = [index for idx, index in enumerate(unfiltEvent_indexes) if idx not in reject_events]
            # event_indexes = unfiltEvent_indexes
                            
            # specify sides
            if 'l_' in segment:
                side = 'Left'
                plot_color = 'blue'
                ax.plot(time.iloc[event_indexes], signal_vel[event_indexes], "kx")
            else:
                side = 'Right'
                plot_color = 'red'
                ax.plot(time.iloc[event_indexes], signal_vel[event_indexes], "kx")
            
            # Plot the results
            ax.plot(time, signal_vel, label=segment, color=plot_color)
            ax.plot(time, rel_segment, label='relative position', color=plot_color, linewidth=0.5, linestyle='--')
            
            ax.set_xlabel("Time")
            ax.set_ylabel("Foot On Events\nForward Velocity")
            # sets the upper left anchor point of the legend to 100% above and right of axis origin
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            # save times and frames to dictionary
            num_events  = len(event_indexes)
            contexts    = [side] * num_events
            labels      = ['Foot Strike'] * num_events
            times       = list(round(time.iloc[event_indexes], 3))
            frames      = list(frame.iloc[event_indexes])
            source      = ['Foot Segment'] * len(labels)
            
            kstr_dict = {
                'Contexts':     contexts,
                'Labels':       labels,
                'Times':        times,
                'Frames':       frames,
                'Source':       source}
            
            df = pd.DataFrame(kstr_dict)
            events_df = pd.concat([events_df, df])
                
        # sort all events
        events_df.sort_values(by='Frames') # ascending order
        
        return events_df
        
    def force_Events(fp_df, threshold):
        fp_cols = [colname for colname in fp_df.columns if colname != 'Time']
        forces_threshold = fp_df[fp_cols] < -threshold
        
        # make sure True flags are present for at least 10 frames/rows
        FSI = np.array(np.where((forces_threshold.shift(10, fill_value=False) == False) & (forces_threshold == True)))
        FOI = np.array(np.where((forces_threshold.shift(10, fill_value=False) == True) & (forces_threshold == False)))
        
        # this will pull the first index from the array of 10 frames where the above criteria is met
        Footstr_indexes = []
        for strIDX in range(0, max(FSI[1]) + 1):
            Footstr_indexes.append(FSI[0][FSI[1] == strIDX].min())
            
        FootOff_indexes = []
        for offIDX in range(0, max(FOI[1]) + 1):
            FootOff_indexes.append(FOI[0][FOI[1] == offIDX].min())
    
        # save times and frames to dictionary
        num_FootOn      = len(Footstr_indexes)
        num_FootOff     = len(FootOff_indexes)
        Onlabels        = ['Foot Strike'] * num_FootOn
        Offlabels       = ['Foot Off'] * num_FootOff
        labels          = Onlabels + Offlabels
        times           = list(round(fp_df['Time'].iloc[Footstr_indexes], 3)) + list(round(fp_df['Time'].iloc[FootOff_indexes], 3))
        fOn             = fp_df['VideoFrames'].iloc[Footstr_indexes]
        fOff            = fp_df['VideoFrames'].iloc[FootOff_indexes]
        frames          = [int(onframe) for onframe in fOn] + [int(offframe) for offframe in fOff]
        source          = ['Force Plate'] * len(labels)
        
        fc_dict = {
            'Labels':       labels,
            'Times':        times,
            'Frames':       frames,
            'Source':       source}
        
        events_force = pd.DataFrame(fc_dict).sort_values(by='Times')
        
        return events_force
    
    def c3d_Events(filenamepath, save_folderpath):
        # --------------------------- Extract C3D data ----------------------------
        c3d = ez(filenamepath)
        parametergroup = c3d['parameters']
        header = c3d['header']
        
        ###########################################################################
        # --------------------------- Pull Events ---------------------------------
        try:
            ev = parametergroup['EVENT'] 
        
            # save times and frames to dictionary
            
            contexts    = ev['CONTEXTS']['value']
            num_events  = ev['USED']['value'][0]
            labels      = ev['LABELS']['value']
            times       = list(ev['TIMES']['value'][1])
            
            # get time of trial
            vfr = header['points']['frame_rate']
            start_frame = header['points']['first_frame']
            end_frame = header['points']['last_frame']    
            time = np.arange(start_frame, end_frame + 1) / vfr
            frame = np.arange(start_frame, end_frame + 1)
            
            # nearest neighbor indexing
            nn_IDX = np.searchsorted(time, times)
            
            # get frames
            frames      = frame[nn_IDX]
            source      = ['Manual ID'] * num_events
            
            mann_dict = {
                'Contexts':     contexts,
                'Labels':       labels,
                'Times':        times,
                'Frames':       frames,
                'Source':       source}
            
            events_man = pd.DataFrame(mann_dict)
        except:
            events_man = pd.DataFrame()
        
        # return events
        return events_man

class segments():
    def get_Segments(filenamepath, save_folderpath, requested_labels, jerk_threshold):
        # --------------------------- Extract C3D data --------------------------------
        filenamepath = 'K:/ViconDatabase/Python Code/Py3_Theia_AutoEventDetect/cd3_test_files/Patien15-07.c3d'
        c3d = ez(filenamepath)
        data = c3d['data']
        parametergroup = c3d['parameters']
        header = c3d['header']
        
        ###########################################################################
        # --------------------------- Pull 4x4 object data ------------------------
        rotations = parametergroup['ROTATION']
        rotdata = data['rotations']
        rotlabels = rotations['LABELS']['value']
        
        # pull segment indexes and lables using supplied lables list    
        flip_directions = ['_x','_y','_z']
        column_names = [f"{name}{suffix}" for name in rotlabels for suffix in flip_directions]
        
        # get position data of segments
        # positions = rotdata[0:3, 3, segments, :]  # shape: (3, 19, 716)
        positions = rotdata[0:3, 3, :, :]
        
        # need to reorder axes: from (3, 4, N) to (N, 4, 3) then reshape
        data_reordered = np.transpose(positions, (2, 1, 0))  # shape becomes (N, 4, 3)
        reshaped = data_reordered.reshape(positions.shape[2], -1)  # now is (N, 4*3)
        
        # segment position dataframe
        segPosition_df = pd.DataFrame(reshaped, columns=column_names)
        
        # -------------------------- Trimming data ----------------------------
        # ------------ TRIMMING WHERE ONLY ALL SEGMENTS ARE PRESENT -----------
        # ensure all segments are present and trim
        # Get indexes of rows that contain any NaNs
        dropped_indexes = segPosition_df[segPosition_df.isna().any(axis=1)].index
        # Drop rows with NaNs from the original DataFrame
        allSeg_df = segPosition_df.drop(index=dropped_indexes)
        
        # keep only needed columns based on requested labels
        keep_columns = []
        for segment in requested_labels:
            for col_name in allSeg_df.columns:
                if segment in col_name:
                    keep_columns.append(col_name)
        
        seg_df = allSeg_df[keep_columns]
        
        # ---------- TRIMMING WHERE SEGMENT JERK ABOVE 95% THRESHOLD ----------
        # calculate segment jerk to find erratic behavior at the beginning and end of trial and trim it off
        seg_jerk = pd.DataFrame()
        get_segs = ['pelvis','l_foot','r_foot','l_toes','r_toes']
        # get pelvis, foot, and toes segments but only left/right at one time
        for seg in get_segs:
            segment_names = [col for col in seg_df.columns if seg in col]
            seg_jerk[seg] = signal_process.compute_jerk(seg_df, segment_names)
            
        # fill nan values with 999 to flag as False - always cropping first 3 rows/frames from differentiation to jerk
        seg_jerk = seg_jerk.fillna(999)
        
        # flag segment jerk values above 95% threshold
        jerk_flags = seg_jerk > np.percentile(seg_jerk, jerk_threshold)
        # only flag rows where ALL columns meet criteria - must be 1D array
        row_flags = np.array(jerk_flags.all(axis=1))
        # only flag at the beginning and end of trial to avoid clipping walking in the middle
        first_valid = np.argmax(~row_flags)  # First False
        last_valid = len(row_flags) - np.argmax(~row_flags[::-1]) - 1  # Last False
        
        # new mask that only keeps True values at the edges
        edge_flags = np.zeros_like(row_flags, dtype=bool)
        edge_flags[:first_valid] = row_flags[:first_valid]  # Keep True flags at the start
        edge_flags[last_valid + 1:] = row_flags[last_valid + 1:]  # Keep True flags at the end
        
        # trim the dataframe using edge_flags
        segment_df = seg_df[~edge_flags].reset_index(drop=True)

        # ------------------ Get trial and frame data -------------------------
        # get video frame rate
        vfr = header['points']['frame_rate']
        
        # get time of trial
        start_frame = header['points']['first_frame']
        end_frame = header['points']['last_frame']  
        t_df = pd.DataFrame()
        t_df['Time'] = np.arange(start_frame, end_frame + 1) / vfr
        t_df['Frame'] = np.arange(start_frame, end_frame + 1)
        
        # trim time dataframe to match trimmed segment dataframe
        t_df = t_df.drop(index=dropped_indexes)
        time_df = t_df[~edge_flags].reset_index(drop=True)
        
        return segment_df, time_df, vfr

    def get_relPosition(seg_df, reference_segment):
        # calculate the relative positions of the foot/toe segments
        reference = seg_df[reference_segment]
        rel_df    = seg_df.drop(columns=reference_segment).sub(reference, axis=0) # > 0
        
        return rel_df
    
    def get_segVelocity(seg_df, rel_df, vfr):
        mrk_vel = seg_df.diff().fillna(0) / vfr
        rel_vel = rel_df.diff().fillna(0) / vfr
        
        # sum z-flip_direction trajectories for each foot then pass to the function to extract foot off events
        mrk_vel['l_foot_z_sum'] = mrk_vel['l_foot_4X4_z'] + mrk_vel['l_toes_4X4_z']
        mrk_vel['r_foot_z_sum'] = mrk_vel['r_foot_4X4_z'] + mrk_vel['r_toes_4X4_z']
        
        rel_vel['l_foot_z_sum'] = rel_vel['l_foot_4X4_z'] + rel_vel['l_toes_4X4_z']
        rel_vel['r_foot_z_sum'] = rel_vel['r_foot_4X4_z'] + rel_vel['r_toes_4X4_z']
        
        # check max absolute value for each trace to help determine if data needs to be flipped
        maxabs, minabs = [mrk_vel.max().abs(), mrk_vel.min().abs()]
        
        if max(maxabs) > max(minabs):
            flip_direction = False
        else:
            # when flip_direction is True, data should be flipped
            flip_direction = True
        
        return mrk_vel, rel_vel, flip_direction

    
    
    
    
    
    
    
    