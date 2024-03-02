import os
from os import path, replace

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed

from collections import defaultdict

#+-----------------------------------static version---------------------+
class AstroDataset(Dataset):
    def __init__(self, im_root, gt_root, num_timestamps, num_frames=50, max_num_obj=1):
        self.im_root = im_root
        self.gt_root = gt_root
        self.num_frames = num_frames
        self.num_slices = num_frames
        self.max_num_obj = max_num_obj
        self.num_timestamps = num_timestamps

        self.im_list = []

        self.videos = []
        self.frames = {} #defaultdict(list)

        self.clips = self._load_clips()

        # Pre-filtering
        for clip_path in self.clips:
            vid_name = self.__get_vid_name(clip_path)
        
            # read all the 50 slices within all timestamps
            frames = self.__read_frames(clip_path)
            
            self.frames[vid_name] = frames
            self.videos.append(vid_name)


        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(self.clips), im_root))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0), # No hue change here as that's not realistic
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9,1.1), shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
            transforms.Resize(384, InterpolationMode.BICUBIC),
            transforms.RandomCrop((384, 384), pad_if_needed=True, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9,1.1), shear=10, interpolation=InterpolationMode.BICUBIC, fill=0),
            transforms.Resize(384, InterpolationMode.NEAREST),
            transforms.RandomCrop((384, 384), pad_if_needed=True, fill=0),
        ])


        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0.05),
            transforms.RandomGrayscale(0.05),
        ])

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=im_mean),
            transforms.RandomHorizontalFlip(),
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=0),
            transforms.RandomHorizontalFlip(),
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            im_normalization,
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
        ])

    def _get_sample(self, frame_folder, case_name, clip_num):
        frame_images = []
        frame_masks = []

        for slice_name in frame_folder:
            jpg_name = slice_name
            png_name = slice_name[:-4] + '.png'

            timestamp = str(int(slice_name.split("_")[-2]))

            # info['frames'].append(jpg_name)

            img_path = os.path.join(self.im_root, case_name, clip_num, timestamp, jpg_name)
            mask_path = os.path.join(self.gt_root, case_name, clip_num, timestamp, png_name)

            sequence_seed = np.random.randint(2147483647)

            reseed(sequence_seed)
            this_im = Image.open(img_path).convert('RGB')       #TODO: change back to 'L'    
            # this_im = self.all_im_dual_transform(this_im)   
            # this_im = self.all_im_lone_transform(this_im)
            # reseed(sequence_seed)
            this_gt = Image.open(mask_path).convert('P')        #TODO: possible change to 'L'
            # this_gt = self.all_gt_dual_transform(this_gt)

            # pairwise_seed = np.random.randint(2147483647)
            # reseed(pairwise_seed)
            # this_im = self.pair_im_dual_transform(this_im)
            # this_im = self.pair_im_lone_transform(this_im)
            # reseed(pairwise_seed)
            # this_gt = self.pair_gt_dual_transform(this_gt)

            # # Use TPS only some of the times
            # # Not because TPS is bad -- just that it is too slow and I need to speed up data loading
            # if np.random.rand() < 0.33:
            #     this_im, this_gt = random_tps_warp(this_im, this_gt, scale=0.02)

            this_im = self.final_im_transform(this_im)
            # this_gt = np.array(this_gt)
            this_gt = self.final_gt_transform(this_gt)

            frame_images.append(this_im)
            frame_masks.append(this_gt)

        # Stack all slices for current timesamp to create a cube
        # images.append(torch.stack(frame_images, 0))
        # masks.append(torch.stack(frame_masks, 0))

        frame_images = torch.stack(frame_images, 0)
        frame_masks = torch.stack(frame_masks, 0)
        
        
        return frame_images, frame_masks.numpy()

    def __getitem__(self, idx):
        video = self.videos[int(idx / self.num_timestamps)]
        case_name = f"SN_{video.split('_')[1]}"      # SN_2016 from SN_2016_0
        clip_num = video.split("_")[-1]     # 0 from SN_2016_0
        info = {}
        info['name'] = video
        info['case_name'] = case_name 
        info['clip_num'] = clip_num

        vid_im_path = path.join(self.im_root, case_name, clip_num)
        vid_gt_path = path.join(self.gt_root, case_name, clip_num)
        frames = self.frames[video]

        additional_objects = np.random.randint(self.max_num_obj)
        indices = [idx, *np.random.randint(self.__len__(), size=additional_objects)]

        merged_images = None
        merged_masks = np.zeros((self.num_frames, 384, 384), dtype=np.int64)

        # # for i, list_id in enumerate(len(frames[video])):      #TODO: was (indices)
        # for frame_id in range(len(frames)):
        #     images, masks = self._get_sample(frames[frame_id], case_name, clip_num)
        #     if merged_images is None:
        #         merged_images = images
        #     else:
        #         merged_images = merged_images*(1-masks) + images*masks
        #     merged_masks[masks[:,0]>0.5] = (frame_id+1)

        frame_id = idx - int(idx / self.num_timestamps) * self.num_timestamps
        images, masks = self._get_sample(frames[frame_id], case_name, clip_num)
        if merged_images is None:
            merged_images = images
        else:
            merged_images = merged_images*(1-masks) + images*masks
        merged_masks[masks[:,0]>0.5] = (frame_id+1)

        #DEBUG
        # print(f"Processing video: {video} frame: {frame_id}")

        # masks = np.stack(masks, 0)
        masks = merged_masks

        # if len(target_objects) > self.max_num_obj:
        #     target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)


        labels = np.unique(masks[0])
        # Remove background
        labels = labels[labels!=0]
        target_objects = labels.tolist()

        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.num_frames, 384, 384), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, 384, 384), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask = (masks==l)
            cls_gt[this_mask] = i+1
            first_frame_gt[0,i] = (this_mask[0])
        

        # Convert cls_gt to a 4D tensor for compatibility: [num_frames, 1, H, W]
        cls_gt = np.expand_dims(cls_gt, axis=1)

        info['num_objects'] = 1                     #TODO: was max(1, len(target_objects))

        # 1 if object exist, 0 otherwise
        # selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = [1 if i < len(np.unique(masks)) - 1 else 0 for i in range(self.max_num_obj)]  # Exclude background
        selector = torch.FloatTensor(selector)

        data = {
            'rgb': merged_images,
            'first_frame_gt': first_frame_gt,
            'cls_gt': cls_gt,
            'selector': selector,
            'info': info
        }


        return data



    def _load_clips(self):
        clips = []
        for movie in sorted(os.listdir(self.im_root)):
            movie_path = os.path.join(self.im_root, movie)
            for clip in sorted(os.listdir(movie_path)):
                clip_path = os.path.join(movie_path, clip)
                clips.append(clip_path)
        return clips

    def __len__(self):
        return sum(len(frame) for frame in self.frames.values())
    
    def __get_vid_name(self, clip_path):
        movie_name = clip_path.split('/')[-2]
        clip_num = clip_path.split('/')[-1]
        return f"{movie_name}_{clip_num}"
    
    def __read_frames(self, clip_path):
        frames = []
        for timestamp in sorted(os.listdir(clip_path)): 
            timestamp_path = os.path.join(clip_path, timestamp)    
            #read all images within time timestamp_path and store into frame
            frames.append(sorted(os.listdir(timestamp_path)))

        # then append the list to frames
        return frames


#+-----------------------------------DAVIS version---------------------+
# class AstroDataset(Dataset):
#     """
#     Works for astro training
#     For each sequence:
#     - Pick 50 frames
#     - Apply some random transforms that are the same for all frames
#     - Apply random transform to each of the frame
#     - The distance between frames is controlled
#     """

#     def __init__(self, im_root, gt_root, max_jump, is_bl, num_slices = 50, subset=None, num_frames=50, max_num_obj=1, finetune=False):
#         self.im_root = im_root
#         self.gt_root = gt_root
#         self.max_jump = max_jump
#         self.is_bl = is_bl
#         self.num_frames = num_frames
#         self.num_slices = num_slices
#         self.max_num_obj = max_num_obj

#         self.videos = []
#         self.frames = {}

#         self.clips = self._load_clips()

#         # vid_list = [self.__get_vid_name(clip_path) for clip_path in self.clips] #sorted(os.listdir(self.im_root))
        
#         # Pre-filtering
#         for clip_path in self.clips:
#             vid_name = self.__get_vid_name(clip_path)
        
#             if subset is not None:      # load the training set names
#                 if vid_name not in subset:   # only load those that's in training set
#                     continue
            
#             # read all the 50 slices within all timestamps
#             frames = self.__read_frames(clip_path)
            
#             self.frames[vid_name] = frames
#             self.videos.append(vid_name)

#         print('%d out of %d videos accepted in %s.' % (len(self.videos), len(self.clips), im_root))

#         # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
#         self.pair_im_lone_transform = transforms.Compose([
#             transforms.ColorJitter(0.01, 0.01, 0.01, 0),
#         ])

#         self.pair_im_dual_transform = transforms.Compose([
#             transforms.RandomAffine(degrees=0 if finetune or self.is_bl else 15, shear=0 if finetune or self.is_bl else 10, interpolation=InterpolationMode.BILINEAR, fill=im_mean),
#         ])

#         self.pair_gt_dual_transform = transforms.Compose([
#             transforms.RandomAffine(degrees=0 if finetune or self.is_bl else 15, shear=0 if finetune or self.is_bl else 10, interpolation=InterpolationMode.NEAREST, fill=0),
#         ])

#         # These transform are the same for all pairs in the sampled sequence
#         self.all_im_lone_transform = transforms.Compose([
#             transforms.ColorJitter(0.1, 0.03, 0.03, 0),
#             transforms.RandomGrayscale(0.05),
#         ])

#         if self.is_bl:
#             # Use a different cropping scheme for the blender dataset because the image size is different
#             self.all_im_dual_transform = transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.BILINEAR)
#             ])

#             self.all_gt_dual_transform = transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.NEAREST)
#             ])
#         else:
#             self.all_im_dual_transform = transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BILINEAR)
#             ])

#             self.all_gt_dual_transform = transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
#             ])

#         # Final transform without randomness
#         self.final_im_transform = transforms.Compose([
#             transforms.ToTensor(),
#             im_normalization,
#         ])

#     def __getitem__(self, idx):
#         video = self.videos[idx]
#         case_name = f"SN_{video.split('_')[1]}"      # SN_2016 from SN_2016_0
#         clip_num = video.split("_")[-1]     # 0 from SN_2016_0
#         info = {}
#         info['name'] = video
#         info['case_name'] = case_name 
#         info['clip_num'] = clip_num

#         vid_im_path = path.join(self.im_root, case_name, clip_num)
#         vid_gt_path = path.join(self.gt_root, case_name, clip_num)
#         frames = self.frames[video]

#         trials = 0
#         while trials < 5:
#             info['frames'] = [] # Appended with actual frames

#             num_frames = self.num_frames
#             length = len(frames)
#             this_max_jump = min(len(frames), self.max_jump)     # check what's max_jump

#             # iterative sampling
#             frames_idx = [np.random.randint(length)]
#             acceptable_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1))).difference(set(frames_idx))
#             while(len(frames_idx) < num_frames):
#                 idx = np.random.choice(list(acceptable_set))
#                 frames_idx.append(idx)
#                 new_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1)))
#                 acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))

#             frames_idx = sorted(frames_idx)
#             if np.random.rand() < 0.5:
#                 # Reverse time
#                 frames_idx = frames_idx[::-1]

#             sequence_seed = np.random.randint(2147483647)
#             images = []
#             masks = []
#             target_objects = []
#             for f_idx in frames_idx:
#                 frame_folder = frames[f_idx]
#                 frame_images = []
#                 frame_masks = []

#                 for slice_name in frame_folder:
#                     jpg_name = slice_name
#                     png_name = slice_name[:-4] + '.png'

#                     info['frames'].append(jpg_name)

#                     img_path = os.path.join(self.im_root, case_name, clip_num, jpg_name)
#                     mask_path = os.path.join(self.gt_root, case_name, clip_num, png_name)

#                     reseed(sequence_seed)
#                     this_im = Image.open(img_path).convert('L')
#                     this_im = self.all_im_dual_transform(this_im)   
#                     this_im = self.all_im_lone_transform(this_im)
#                     reseed(sequence_seed)
#                     this_gt = Image.open(mask_path).convert('P')        #TODO: possible change to 'L'
#                     this_gt = self.all_gt_dual_transform(this_gt)

#                     pairwise_seed = np.random.randint(2147483647)
#                     reseed(pairwise_seed)
#                     this_im = self.pair_im_dual_transform(this_im)
#                     this_im = self.pair_im_lone_transform(this_im)
#                     reseed(pairwise_seed)
#                     this_gt = self.pair_gt_dual_transform(this_gt)

#                     this_im = self.final_im_transform(this_im)
#                     this_gt = np.array(this_gt)

#                     frame_images.append(this_im)
#                     frame_masks.append(this_gt)

#                 # Stack all slices for current timesamp to create a cube
#                 images.append(torch.stack(frame_images, 0))
#                 masks.append(torch.stack(frame_masks, 0))
            
#             labels = np.unique(masks[0][0])     #DEBUG might lead to dimention error? remove if trigger sth
#             # Remove background
#             labels = labels[labels!=0]

#             if self.is_bl:
#                 # Find large enough labels
#                 good_lables = []
#                 for l in labels:
#                     pixel_sum = (masks[0]==l).sum()
#                     if pixel_sum > 10*10:
#                         # OK if the object is always this small
#                         # Not OK if it is actually much bigger
#                         if pixel_sum > 30*30:
#                             good_lables.append(l)
#                         elif max((masks[1]==l).sum(), (masks[2]==l).sum()) < 20*20:
#                             good_lables.append(l)
#                 labels = np.array(good_lables, dtype=np.uint8)
            
#             if len(labels) == 0:
#                 target_objects = []
#                 trials += 1
#             else:
#                 target_objects = labels.tolist()
#                 break

#         if len(target_objects) > self.max_num_obj:
#             target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)

#         info['num_objects'] = max(1, len(target_objects))

#         masks = np.stack(masks, 0)

#         # Generate one-hot ground-truth
#         cls_gt = np.zeros((self.num_frames, self.num_slices, 384, 384), dtype=np.int64)
#         first_frame_gt = np.zeros((1, self.max_num_obj, 384, 384), dtype=np.int64)

#         for frame_idx in range(self.num_frames):
#             for slice_idx in range(self.num_slices):
#                 labels = np.unique(masks[frame_idx][slice_idx])
#                 labels = labels[labels != 0]  # Remove background

#                 for i, label in enumerate(labels):
#                     label_mask = (masks[frame_idx][slice_idx] == label)

#                     cls_gt[frame_idx, slice_idx, label_mask] = i + 1

#                     # For the first frame, set the ground truth for the first occurrence of each object
#                     if frame_idx == 0:
#                         first_frame_gt[0, i, label_mask] = 1

#         # Convert cls_gt to a 5D tensor for compatibility: [num_frames, 1, num_slices, H, W]
#         cls_gt = np.expand_dims(cls_gt, axis=1)

#         # 1 if object exist, 0 otherwise
#         # selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
#         selector = [1 if i < len(np.unique(masks)) - 1 else 0 for i in range(self.max_num_obj)]  # Exclude background
#         selector = torch.FloatTensor(selector)

#         data = {
#             'rgb': images,
#             'first_frame_gt': first_frame_gt,
#             'cls_gt': cls_gt,
#             'selector': selector,
#             'info': info,
#         }

#         return data

#     def _load_clips(self):
#         clips = []
#         for movie in sorted(os.listdir(self.im_root)):
#             movie_path = os.path.join(self.im_root, movie)
#             for clip in sorted(os.listdir(movie_path)):
#                 clip_path = os.path.join(movie_path, clip)
#                 clips.append(clip_path)
#         return clips

#     def __len__(self):
#         return len(self.videos)
    
#     def __get_vid_name(self, clip_path):
#         movie_name = clip_path.split('/')[-2]
#         clip_num = clip_path.split('/')[-1]
#         return f"{movie_name}_{clip_num}"
    
#     def __read_frames(self, clip_path):
#         for timestamp in sorted(os.listdir(clip_path)):
#             frame_images = []
#             timestamp_path = os.path.join(clip_path, timestamp)     #TODO: might need to convert to str
#             #read all images within time timestamp_path and store into frame
#             frame_images.append(sorted(os.listdir(timestamp_path)))

#         # then append the list to frames
#         return frame_images




