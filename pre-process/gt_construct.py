import cv2
import numpy as np
import pandas as pd
import glob
import os
import argparse
from utils import *

low_x0, low_y0, low_w, low_h, bottom_z, top_z = -500, -500, 1000, 1000, -500, 500


def associate_slices_within_cube(start_z, end_z, image_paths, mask, dataset_root, SN_timestamp, timestamp, i, direction, date):
    tmp_mask = mask

    for img_path_id in range(start_z, end_z, direction):    # direction: +1 tracking down, -1 tracking up

        image_path = image_paths[img_path_id]
        image = read_image_grayscale(image_path)
        binary_image = apply_otsus_thresholding(image)
        num_labels, labels, stats, centroids = find_connected_components(binary_image)

        no_match = True

        for label in range(2, num_labels):
            current_mask = labels == label
            if compute_iou(current_mask, tmp_mask) >= 0.6:      # if found a match in this slice
                tmp_mask = current_mask
                mask_dir_root = ensure_dir(os.path.join(dataset_root, f'SN_cases_{date}', f"SN_{SN_timestamp}{i}", str(timestamp)))
                mask_name = f"{image_path.split('/')[-1].split('.')[-2]}.png"     
                cv2.imwrite(os.path.join(mask_dir_root, mask_name), current_mask * 255)
                no_match = False
                break       # Moving to the next slice
        
        # If can't find any match in this slice, then move on to the next phase
        if no_match:
            break


def trace_current_timestamp(mask_candidates, timestamp, image_paths, all_data, dataset_root, date, incr_interval):
    # filter out all the SN events
    time_Myr = timestamp2time_Myr(timestamp)
    filtered_data = filter_data(all_data[(all_data['time_Myr'] >= time_Myr - (incr_interval / 10)) & (all_data['time_Myr'] < time_Myr)],
                            (low_x0, low_y0, low_w, low_h, bottom_z, top_z))

    for SN_num in range(filtered_data.shape[0]):
        posx_pc = int(filtered_data.iloc[SN_num]["posx_pc"])
        posy_pc = int(filtered_data.iloc[SN_num]["posy_pc"])
        posz_pc = int(filtered_data.iloc[SN_num]["posz_pc"])

        center_slice_z = pc2pixel(posz_pc, x_y_z = "z")

        anchor_img = read_image_grayscale(image_paths[center_slice_z])
        binary_image = apply_otsus_thresholding(anchor_img)
        num_labels, labels, stats, centroids = find_connected_components(binary_image)

        for i in range(2, num_labels):     
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 1000:
                continue
            

            x1 = stats[i, cv2.CC_STAT_LEFT] 
            y1 = stats[i, cv2.CC_STAT_TOP] 
            w = stats[i, cv2.CC_STAT_WIDTH] 
            h = stats[i, cv2.CC_STAT_HEIGHT] 

            # tracing this SN in the bubble for the entire cube 
            if SN_center_in_bubble(pc2pixel(posx_pc, x_y_z = "x"), pc2pixel(posy_pc, x_y_z = "y"), x1, y1, w, h):
                # it is a new SN case, construct a new profile for the SN case
                mask = labels == i
                if not in_mask_candidates(mask_candidates, mask):
                    mask_candidates.append(mask)

                    # then it should save the mask to the new mask folder
                    mask_dir_root = ensure_dir(os.path.join(dataset_root, f'SN_cases_{date}', f"SN_{timestamp}{i}"))
                    mask_name = f"{image_paths[0].split('/')[-1].split('.')[-2]}.png"     
                    cv2.imwrite(os.path.join(mask_dir_root, str(timestamp), mask_name), mask * 255)
                    with open(os.path.join(mask_dir_root, f"SN_{timestamp}{i}_info.txt"), "w") as f:
                        f.write(str(filtered_data.iloc[SN_num]))

                    # tracking up       
                    associate_slices_within_cube(center_slice_z - 1, 0, image_paths, mask, dataset_root, timestamp, timestamp, i, -1, date)
                    # tracking down
                    associate_slices_within_cube(center_slice_z, 1000, image_paths, mask, dataset_root, timestamp, timestamp, i, 1, date)
                
    return mask_candidates
            



def associate_subsequent_timestamp(timestamp, start_timestamp, end_timestamp, incr, dataset_root, date, raw_img_format):
    # loop through all slices in the mask folder
    img_prefix = "sn34_smd132_bx5_pe300_hdf5_plt_cnt_0"
    
    SN_ids = retrieve_id(glob.glob(os.path.join(dataset_root, f'SN_cases_{date}', f'SN_{timestamp}*'))) 
    
    for SN_id in SN_ids:  
        center_z = pc2pixel(read_info(os.path.join(dataset_root, f'SN_cases_{date}', f"SN_{timestamp}{SN_id}", f"SN_{timestamp}{SN_id}_info.txt"), info_col = "posz_pc"), x_y_z = "z")
        # ignore cases if there's no info file
        if center_z == top_z:
            continue
        
        mask = read_image_grayscale(os.path.join(dataset_root, f'SN_cases_{date}', f"SN_{timestamp}{SN_id}", str(timestamp), f"{img_prefix}{timestamp}_z{center_z}.png"))


        for time in range(start_timestamp, end_timestamp, incr):
            # Debug
            print(f"Associating case SN_{timestamp}{SN_id}: {time}")


            image_paths = sort_image_paths(glob.glob(os.path.join(dataset_root, 'raw_img', str(time), f"{img_prefix}{time}_z*.{raw_img_format}"))) 

            # tracking up
            associate_slices_within_cube(center_z - 1, 0, image_paths, mask, dataset_root, timestamp, time, SN_id, -1, date)
            # tracking down
            associate_slices_within_cube(center_z, 1000, image_paths, mask, dataset_root, timestamp, time, SN_id, 1, date)
    


def read_dat_log(dat_file_root, dataset_root):
    dat_files = [os.path.join(dataset_root, dat_file_root, "SNfeedback.dat")]
    
    # Initialize an empty DataFrame
    all_data = pd.DataFrame()

    # Read and concatenate data from all .dat files
    for dat_file in dat_files:
        # Assuming space-separated values in the .dat files
        df = pd.read_csv(dat_file, delim_whitespace=True, header=None,
                        names=['n_SN', 'type', 'n_timestep', 'n_tracer', 'time',
                                'posx', 'posy', 'posz', 'radius', 'mass'])
        
        # Convert the columns to numerical
        df = df.iloc[1:]
        df['n_SN'] = df['n_SN'].map(int)
        df['type'] = df['type'].map(int)
        df['n_timestep'] = df['n_timestep'].map(int)
        df['n_tracer'] = df['n_tracer'].map(int)
        df['time'] = pd.to_numeric(df['time'],errors='coerce')
        df['posx'] = pd.to_numeric(df['posx'],errors='coerce')
        df['posy'] = pd.to_numeric(df['posy'],errors='coerce')
        df['posz'] = pd.to_numeric(df['posz'],errors='coerce')
        df['radius'] = pd.to_numeric(df['radius'],errors='coerce')
        df['mass'] = pd.to_numeric(df['mass'],errors='coerce')
        all_data = pd.concat([all_data, df], ignore_index=True)
        all_data = all_data.drop(df[df['n_tracer'] != 0].index)

    # Convert time to Megayears
    all_data['time_Myr'] = seconds_to_megayears(all_data['time'])

    # Convert 'pos' from centimeters to parsecs
    all_data['posx_pc'] = cm2pc(all_data['posx'])
    all_data['posy_pc'] = cm2pc(all_data['posy'])
    all_data['posz_pc'] = cm2pc(all_data['posz'])

    # Sort the DataFrame by time in ascending order
    all_data.sort_values(by='time_Myr', inplace=True)

    return all_data



def main(args):
    timestamps = range(int(args.start_timestamp), int(args.end_timestamp), args.incr)  # List of timestamps
    

    # Read and process the .dat file logs
    all_data_df = read_dat_log(args.dat_file_root, args.dataset_root)

    for timestamp in timestamps:
        # Refresh mask candidates, so that we can record those SN events that happen in the same blob at a later time (move out of the for loop if not wanna record repeating blobs anymore)
        mask_candidates = []

        image_paths = sort_image_paths(glob.glob(os.path.join(args.dataset_root, 'raw_img', str(timestamp), f'*.{args.raw_img_format}'))) # List of image paths for this timestamp


        # DEBUG 
        print(f"\n\nStart tracing through time {timestamp}")


        mask_candidates = trace_current_timestamp(mask_candidates, timestamp, image_paths, all_data_df, args.dataset_root, args.date, args.incr)

        # DEBUG 
        print(f"Done tracing through time {timestamp}")
        print("Start associating with the subsequent timestamp")

        # associate with all later timestamp
        associate_subsequent_timestamp(timestamp, timestamp + args.incr, int(args.end_timestamp), args.incr, args.dataset_root, args.date, args.raw_img_format)
        
        # DEBUG
        print(f"Done associating for timestamp {timestamp}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_timestamp", help="The starting timestamp for the dataset", type = int)             
    parser.add_argument("--end_timestamp", help="The ending timestamp for the dataset", type = int)               
    parser.add_argument("--incr", help="The timestamp increment interval", default = 1, type = int) 
    parser.add_argument("--dataset_root", help="The root directory to the dataset")         
    parser.add_argument("--dat_file_root", help="The root directory to the SNfeedback files, relative to dataset root")         
    parser.add_argument("--date", help="Enter today's date in mmdd format")
    parser.add_argument("--raw_img_format", help=".png or .jpg for raw images", default="jpg")

    # python data/gt_construct.py --start_timestamp 200 --end_timestamp 219 --incr 10 --dataset_root "../../Dataset" --dat_file_root "SNfeedback" --date 0122 --img_extension "jpg" > output.txt 2>&1 &
    
    args = parser.parse_args()
    main(args)