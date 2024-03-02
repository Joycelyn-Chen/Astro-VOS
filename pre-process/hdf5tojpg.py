import yt
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2 as cv

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def main(args):
    for i in range(args.start_Myr, args.end_Myr, args.offset):
        if (i < 1000):
            filename = f"{args.file_prefix}0{i}"
        else:
            filename = f"{args.file_prefix}{i}"

        # loading img data
        ds = yt.load(os.path.join(args.input_dir, filename))
        #ds.current_time, ds.current_time.to('Myr')
        # ad = ds.all_data()

        center =  [0, 0, 0]*yt.units.pc
        arb_center = ds.arr(center,'code_length')
        left_edge = arb_center - ds.quan(int(args.xlim / 2),'pc')
        right_edge = arb_center + ds.quan(int(args.ylim / 2),'pc')
        obj = ds.arbitrary_grid(left_edge, right_edge, dims=[args.xlim, args.ylim, args.zlim])

        # Saving img
        for j in range(int(args.zlim)):
            img = np.log10(obj["flash", "dens"][:,:,j].T[::-1])
            normalizedImg = ((img - np.min(img)) / (np.max(img) - np.min(img)) ) * 255 

            cv.imwrite(os.path.join(ensure_dir(os.path.join(args.output_root_dir, str(i))), f'{filename}_z{j}{args.extension}'), normalizedImg)
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="The root directory for the hdf5 dataset")              
    parser.add_argument("--output_root_dir", help="The root directory for the img output")          
    parser.add_argument("--file_prefix", help="file prefix", default="sn34_smd132_bx5_pe300_hdf5_plt_cnt_")        
    parser.add_argument("--start_Myr", help="The starting Myr for data range", type = int)          
    parser.add_argument("--end_Myr", help="The end Myr for data range", type = int)                 
    parser.add_argument("--offset", help="The offset for incrementing Myr", type = int, default = 1)                
    parser.add_argument("--xlim", help="Input xlim", type = int, default = 1000)                                          
    parser.add_argument("--ylim", help="Input ylim", type = int, default = 1000)                                         
    parser.add_argument("--zlim", help="Input zlim", type = int, default = 1000)                                         
    parser.add_argument("--extension", help="Input the image extension (.jpg, .png)", default=".jpg")       

    args = parser.parse_args()
    main(args)