import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dust3r.dummy_io import *
os.environ["meta_internal"] = "False"

import copy
from copy import deepcopy

import numpy as np
import re
import torch
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation

from dust3r.inference import inference_mv
from dust3r.losses import calibrate_camera_pnpransac, estimate_focal_knowing_depth
from dust3r.model import AsymmetricCroCo3DStereoMultiView
from dust3r.utils.device import to_numpy

from dust3r.utils.image import load_images
from dust3r.viz import add_scene_cam, CAM_COLORS, cat_meshes, OPENGL, pts3d_to_trimesh

import root_file_io as fio
import random
import string
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1
inf = np.inf

def select_random_chunk(my_list, min_size=12, max_size=24, step=1, start_index=None):
    if step < 1:
        step = 1
    selection_size = random.randint(min_size, max_size)
    max_start_index = len(my_list) - (selection_size - 1) * step
    if max_start_index <= 0:
        max_start_index = len(my_list) - 1
    if start_index is None:
        start_index = random.randint(0, max_start_index)
    random_indices = list(range(start_index, start_index + selection_size * step, step))
    random_subset = [my_list[i] for i in random_indices if i < len(my_list)]
    return random_subset, random_indices


# Function to extract the number from the folder name
def extract_number_foldername(folder_path):
    # Extract the folder name (assuming it's the last part of the path)
    folder_name = folder_path.split("/")[-1]
    # Use regex to find the numeric part of the folder name
    match = re.search(r'\d+', folder_name)
    return int(match.group()) if match else float('inf')  # Return infinity if no number is found


def extract_number_filename(file_path):
    # Use regular expression to find the numeric part
    match = re.search(r'\d+', file_path)
    return int(match.group()) if match else 0


class CSlamChunk:
    def __init__(self, scene_dir, index):
        self.identity = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        self.scene_dir = scene_dir 
        fio.ensure_dir(self.scene_dir)
        self.__ply_path = fio.createPath(fio.sep, [scene_dir], "model.ply")
        self.__glb_path = fio.createPath(fio.sep, [scene_dir], "model.glb")
        self.global_index = index
        self.parent_index = -1
        self.tags = ['pts3d', 'msk', 'focals', 'cams2world', 'intrinsics', 'images', '3d_models']


    def set_source_images(self, file_lists):
        self.__source_images = file_lists

    def get_source_images(self):
        return self.__source_images
    
    def get_ply_save_path(self):
        return self.__ply_path
    
    def get_glb_save_path(self):
        return self.__glb_path
    
    def get_config_path(self, tag=None):
        rslt = ''
        if tag == None:
            rslt = self.scene_dir
            fio.ensure_dir(rslt)
        elif tag == 'intrinsics':
            fio.ensure_dir(self.scene_dir)
            rslt = os.path.join(self.scene_dir, tag)
        else:
            rslt = os.path.join(self.scene_dir, tag)
            fio.ensure_dir(rslt)
        return rslt

    def save_scene(self, rgbimg, pts3d, msk, cams2world, focals, intrinsics, index=-1):
        np.save(os.path.join(self.get_config_path(tag='intrinsics')) + '.npy', intrinsics.cpu())
        # self.convert_scene_output_to_ply(os.path.join(self.get_config_path(tag='3d_models')), rgbimg, pts3d, msk)
        self.convert_scene_output_to_glb(os.path.join(self.get_config_path(tag='mesh')), rgbimg, pts3d, msk, focals, cams2world, index=index)

    def load_rgb_images(self, rgb_dir):
        from glob import glob
        image_paths = sorted(glob(os.path.join(rgb_dir, "*.png")))
        rgb_images = [np.array(Image.open(img_path)) for img_path in image_paths]
        return rgb_images
    
    def load_config_data(self):
        data = {}
        data["intrinsics"] = np.load(self.get_config_path(tag='intrinsics') + '.npy', allow_pickle=True)
        # data['ply_path'] = os.path.join(self.get_config_path(tag='3d_models'), f"scene.ply")
        # data['glb_path'] = os.path.join(self.get_config_path(tag='3d_models'), f"scene.glb")
        data["content"] = []
        image_dir = self.get_config_path(tag='images')
        image_paths = fio.traverse_dir(image_dir, full_path=True, towards_sub=False)
        image_paths = fio.filter_ext(image_paths, filter_out_target=False, ext_set=fio.img_ext_set)
        image_paths = sorted(
            image_paths, 
            key=lambda path: int(path.split('/')[-1].split('_')[1].replace('.png', ''))
        )

        for i, img_pth in enumerate(image_paths):
            sub_dict = {}
            sub_image = Image.open(img_pth)
            img_array = np.array(sub_image)
            img_float = img_array.astype(np.float32) / 255.0 
            sub_pts3d = np.load(os.path.join(self.get_config_path(tag='pts3d'), f"pts3d_{i}.npy"), allow_pickle=True)
            sub_msk = np.load(os.path.join(self.get_config_path(tag='msk'), f"msk_{i}.npy"), allow_pickle=True)
            sub_focal = np.load(os.path.join(self.get_config_path(tag='focals'), f"focals_{i}.npy"), allow_pickle=True)
            sub_cams2world = np.load(os.path.join(self.get_config_path(tag='cams2world'), f"cams2world_{i}.npy"), allow_pickle=True)
            sub_dict['pts3d'] = sub_pts3d
            sub_dict['msk'] = sub_msk
            sub_dict['focals'] = sub_focal
            sub_dict['cams2world'] = sub_cams2world
            sub_dict['images'] = img_float
            data["content"].append(sub_dict)
        return data
    
    def invert_cams2world(self, cams2world):
        # Invert the rotation matrix
        R = cams2world[:3, :3]
        R_inv = R.T  # Transpose of the rotation matrix

        # Invert the translation vector
        t = cams2world[:3, 3]
        t_inv = -R_inv @ t  # Negate and apply the inverted rotation

        # Construct the world-to-camera matrix
        world2cams = np.eye(4)
        world2cams[:3, :3] = R_inv
        world2cams[:3, 3] = t_inv

        return world2cams

    def convert_scene_output_to_ply(self, outdir, imgs, pts3d, mask):
        """
        Converts the scene to a 3D Gaussian splatting model and saves it as a .ply file.

        Args:
            outdir (str): Output directory for the .ply file.
            imgs (list): List of RGB images corresponding to the cameras.
            pts3d (list): List of 3D points for each camera.
            mask (list): List of masks to filter valid points.
            silent (bool): If True, suppress print statements.

        Returns:
            str: Path to the generated .ply file.
        """
        assert len(pts3d) == len(mask) <= len(imgs)

        # Convert input to numpy arrays
        pts3d = to_numpy(pts3d)
        imgs = to_numpy(imgs)

        # Combine point clouds and colors using the masks
        points = np.concatenate([p[m] for p, m in zip(pts3d, mask)], axis=0)  # Shape: (N, 3)
        colors = np.concatenate([p[m] for p, m in zip(imgs, mask)], axis=0)   # Shape: (N, 3)

        # Ensure colors are in the range [0, 255]
        if colors.max() <= 1.0:  # If colors are normalized, scale them to 0-255
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

        # Construct the PLY file data
        num_points = points.shape[0]
        ply_header = f"""ply
        format ascii 1.0
        element vertex {num_points}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        """
        # Combine points and colors for saving
        ply_data = np.hstack([points, colors])  # Shape: (N, 6)

        # Save to .ply file
        outfile = os.path.join(outdir, 'scene.ply')
        with open(outfile, 'w') as f:
            f.write(ply_header)
            np.savetxt(f, ply_data, fmt='%f %f %f %d %d %d')
        return outfile
    
    def convert_scene_output_to_ply_1(self, outdir, imgs, pts3d, mask, cams2worlds):
        """
        Converts the scene to a 3D Gaussian splatting model and saves it as a .ply file.
        Now includes per-view transformation using the provided `cams2worlds` matrices.

        Args:
            outdir (str): Output directory for the .ply file.
            imgs (list): List of RGB images corresponding to the cameras.
            pts3d (list): List of 3D points for each camera (each as an Nx3 numpy array).
            mask (list): List of masks (boolean arrays) indicating valid points for each camera.
            cams2worlds (list): List of 4x4 transformation matrices, one per camera/view.
        
        Returns:
            str: Path to the generated .ply file.
        """
        # assert len(pts3d) == len(mask) == len(imgs) == len(cams2worlds)

        transformed_points_list = []
        transformed_colors_list = []

        for p, m, T, img in zip(pts3d, mask, cams2worlds, imgs):
            # Filter valid points and corresponding colors
            valid_points = p[m]  # Shape: (N, 3)
            valid_colors = img[m]  # Assuming img is in a compatible shape (N, 3) for colors

            # Convert valid points to homogeneous coordinates
            ones = np.ones((valid_points.shape[0], 1), dtype=valid_points.dtype)
            valid_points_hom = np.concatenate([valid_points, ones], axis=1)  # Shape: (N, 4)

            # Apply the transformation matrix T (4x4)
            transformed_hom = (T @ valid_points_hom.T).T  # Shape: (N, 4)

            # Convert back to 3D coordinates (assuming no projective scaling)
            transformed_points = transformed_hom[:, :3]
            transformed_points_list.append(transformed_points)
            transformed_colors_list.append(valid_colors)

        # Combine points and colors for saving
        points = np.concatenate(transformed_points_list, axis=0)  # Shape: (Total_N, 3)
        colors = np.concatenate(transformed_colors_list, axis=0)    # Shape: (Total_N, 3)

        # Ensure colors are in the range [0, 255]
        if colors.max() <= 1.0:  # If colors are normalized, scale them to 0-255
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

        num_points = points.shape[0]
        ply_header = f"""ply
            format ascii 1.0
            element vertex {num_points}
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
        """

        # Combine points and colors for saving in the PLY file
        ply_data = np.hstack([points, colors])  # Shape: (Total_N, 6)
        outfile = os.path.join(outdir, 'scene.ply')
        with open(outfile, 'w') as f:
            f.write(ply_header)
            np.savetxt(f, ply_data, fmt='%f %f %f %d %d %d')
        return outfile

    def convert_scene_output_to_glb(self, outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                    cam_color=None,
                                    transparent_cams=False, silent=False, 
                                    index=-1):
        assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
        pts3d = to_numpy(pts3d)
        imgs = to_numpy(imgs)
        focals = to_numpy(focals)
        cams2world = to_numpy(cams2world)

        scene = trimesh.Scene()
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

        # add each camera
        # for i, pose_c2w in enumerate(cams2world):
        #     if isinstance(cam_color, list):
        #         camera_edge_color = cam_color[i]
        #     else:
        #         camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        #     add_scene_cam(scene, pose_c2w, camera_edge_color,
        #                 None if transparent_cams else imgs[i], focals[i],
        #                 imsize=imgs[i].shape[1::-1], screen_width=cam_size)

        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
        scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
        outfile = os.path.join(outdir, f'{str(index)}.glb')
        if not silent:
            print('(exporting 3D scene to', outfile, ')')
        scene.export(file_obj=outfile)

    def load_ply(self, ply_path):
        point_cloud = o3d.io.read_point_cloud(ply_path)
        return point_cloud

class CSlamPredictor:
    # Constructor to initialize variables
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.scenes = []
        self.scenes_info = []

    def combine_scenes(self):
        global_scene_chunk = None
        last_frame = None
        pcds = []
        for i, scene in enumerate(self.scenes):
            scene_chunk_info = self.scenes_info[i]
            scene_chunk = scene_chunk_info.load_config_data()
            if i == 0:
                global_scene_chunk = scene_chunk
                last_frame = global_scene_chunk['content'][-1]
                rgbimg, pts3d, msk, focals, cams2world, intrinsics= self.get_3D_model_from_scene(output=scene)
                temp_cache_path = scene_chunk_info.convert_scene_output_to_ply('temp', rgbimg, pts3d, msk)
                pcd = o3d.io.read_point_cloud(temp_cache_path)
                pcds.append(pcd)
                fio.delete_file(temp_cache_path)
            else:
                last_cams2world = last_frame['cams2world']
                current_frame = scene_chunk['content'][0]
                current_cams2world = current_frame['cams2world']
                T_global = last_cams2world @ np.linalg.inv(current_cams2world)

                last_world2cams = self.invert_cams2world(last_cams2world)
                current_world2cams = self.invert_cams2world(current_cams2world)
                # T_global_w2c = last_world2cams @ np.linalg.inv(current_world2cams)
                T_global_w2c = np.linalg.inv(last_world2cams) @ current_world2cams

                cam2world_path = scene_chunk_info.get_config_path('cams2world')
                fio.copy_folder(cam2world_path, cam2world_path + '_original')
                for j, frame in enumerate(scene_chunk['content']):
                    cams2world = frame['cams2world']
                    # cams2world_transformed = T_global @ cams2world 
                    world2cam = self.invert_cams2world(cams2world=cams2world)
                    world2cam_transformed = T_global_w2c @ world2cam
                    cams2world_transformed = self.invert_world2cams(world2cam_transformed)
                    
                    file_path = os.path.join(scene_chunk_info.get_config_path(tag='cams2world'), f"cams2world_{j}.npy")
                    fio.delete_file(file_path)
                    np.save(file_path, cams2world_transformed)

        test_c2w = []
        test_rgb = []
        test_pts3d = []
        test_focal = []
        test_msk = []
        for i, scene in enumerate(self.scenes):
            scene_chunk_info = self.scenes_info[i]
            scene_chunk = scene_chunk_info.load_config_data()

            for j, frame in enumerate(scene_chunk['content']):
                    cams2world = frame['cams2world']
                    rgbimg = frame['images']
                    pts3d = frame['pts3d']
                    msk = frame['msk']
                    focals = frame['focals']
                    test_c2w.append(cams2world)
                    test_rgb.append(rgbimg)
                    test_pts3d.append(pts3d)
                    test_focal.append(focals)
                    test_msk.append(msk)

        self.convert_scene_output_to_glb('temp', 
                    test_rgb, test_pts3d, test_msk, test_focal, test_c2w)


    def extract_rotation_translation(self, cams2world):
        R = cams2world[:3, :3]
        # Extract translation vector (top-right 3x1 column)
        t = cams2world[:3, 3]
        return R, t

    def invert_cams2world(self, cams2world):

        # Invert the rotation matrix
        R = cams2world[:3, :3]
        R_inv = R.T  # Transpose of the rotation matrix

        # Invert the translation vector
        t = cams2world[:3, 3]
        t_inv = -R_inv @ t  # Negate and apply the inverted rotation

        # Construct the world-to-camera matrix
        world2cams = np.eye(4)
        world2cams[:3, :3] = R_inv
        world2cams[:3, 3] = t_inv

        return world2cams
    
    def invert_world2cams(self, world2cams):
        # Extract the rotation matrix and translation vector
        R = world2cams[:3, :3]  # Rotation matrix
        t = world2cams[:3, 3]   # Translation vector

        # Invert the rotation matrix (transpose)
        R_inv = R.T  # Rotation matrix for cams2world

        # Compute the translation vector for cams2world
        t_inv = -R_inv @ t  # Translation vector for cams2world

        # Construct the cams2world matrix
        cams2world = np.eye(4)
        cams2world[:3, :3] = R_inv
        cams2world[:3, 3] = t_inv

        return cams2world


    def generate_new_scene(self, file_list, save_dir, device='cuda', image_size=224, min_conf_thr=3, transparent_cams=True, cam_size=0.05, index=-1):
        scene = self.get_reconstructed_scene(filelist=file_list, device=device, image_size=image_size, n_frame=len(file_list))
        self.save_scene_info(scene, save_dir=save_dir, file_list=file_list, min_conf_thr=min_conf_thr, transparent_cams=transparent_cams, cam_size=cam_size, index=index)

    def save_scene_info(self, scene, save_dir, file_list, min_conf_thr=3, transparent_cams=True, cam_size=0.05, index=-1):
        current_index = len(self.scenes)
        current_chunk= CSlamChunk(scene_dir=save_dir, index=current_index)
        current_chunk.set_source_images(file_lists=file_list)
    
        rgbimg, pts3d, msk, focals, cams2world, intrinsics= self.get_3D_model_from_scene(
            output=scene, min_conf_thr=min_conf_thr, transparent_cams=transparent_cams, cam_size=cam_size)
        current_chunk.save_scene(rgbimg=rgbimg, pts3d=pts3d, msk=msk, cams2world=cams2world, focals=focals, intrinsics=intrinsics, index=index)
        return current_chunk

    def get_reconstructed_scene(self, device, image_size, filelist, n_frame, silent=True):
        """
        from a list of images, run dust3r inference, global aligner.
        """
        imgs = load_images(filelist, size=image_size, verbose=not silent, n_frame = n_frame)
        if len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1
        for img in imgs:
            img['true_shape'] = torch.from_numpy(img['true_shape']).long()

        if len(imgs) < 12:
            if len(imgs) > 3:
                imgs[1], imgs[3] = deepcopy(imgs[3]), deepcopy(imgs[1])
            if len(imgs) > 6:
                imgs[2], imgs[6] = deepcopy(imgs[6]), deepcopy(imgs[2])
        else:
            change_id = len(imgs) // 4 + 1
            imgs[1], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[1])
            change_id = (len(imgs) * 2) // 4 + 1
            imgs[2], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[2])
            change_id = (len(imgs) * 3) // 4 + 1
            imgs[3], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[3])
        
        output = inference_mv(imgs, model, device, verbose=False)
        # ['pts3d', 'conf', 'rgb', 'opacity', 'scale', 'rotation']
                # (['conf', 'rgb', 'opacity', 'scale', 'rotation', 'pts3d_in_other_view'])
        output['pred1']['rgb'] = imgs[0]['img'].permute(0,2,3,1)
        for x, img in zip(output['pred2s'], imgs[1:]):
            x['rgb'] = img['img'].permute(0,2,3,1)
        return output
    
    def get_3D_model_from_scene(self, output, min_conf_thr=3, transparent_cams=False, cam_size=0.05, silent=False):
        with torch.no_grad():
            _, h, w = output['pred1']['rgb'].shape[0:3] # [1, H, W, 3]
            rgbimg = [output['pred1']['rgb'][0]] + [x['rgb'][0] for x in output['pred2s']]
            for i in range(len(rgbimg)):
                rgbimg[i] = (rgbimg[i] + 1) / 2
            pts3d = [output['pred1']['pts3d'][0]] + [x['pts3d_in_other_view'][0] for x in output['pred2s']]
            conf = torch.stack([output['pred1']['conf'][0]] + [x['conf'][0] for x in output['pred2s']], 0)
            conf_sorted = conf.reshape(-1).sort()[0]
            conf_thres = conf_sorted[int(conf_sorted.shape[0] * float(min_conf_thr) * 0.01)]
            msk = conf >= conf_thres
            
            # calculate focus:
            conf_first = conf[0].reshape(-1) # [bs, H * W]
            conf_sorted = conf_first.sort()[0] # [bs, h * w]
            conf_thres = conf_sorted[int(conf_first.shape[0] * 0.03)]
            valid_first = (conf_first >= conf_thres) # & valids[0].reshape(bs, -1)
            valid_first = valid_first.reshape(h, w)
            focals = estimate_focal_knowing_depth(pts3d[0][None].cuda(), valid_first[None].cuda()).cpu().item()

            intrinsics = torch.eye(3,)
            intrinsics[0, 0] = focals
            intrinsics[1, 1] = focals
            intrinsics[0, 2] = w / 2
            intrinsics[1, 2] = h / 2
            intrinsics = intrinsics.cuda()

            focals = torch.Tensor([focals]).reshape(1,).repeat(len(rgbimg))

             # Generate extrinsics (camera poses)
            y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float().cuda() # [H, W, 2]
            
            c2ws = []
            for (pr_pt, valid) in zip(pts3d, msk):
                c2ws_i = calibrate_camera_pnpransac(pr_pt.cuda().flatten(0,1)[None], pixel_coords.flatten(0,1)[None], valid.cuda().flatten(0,1)[None], intrinsics[None])
                c2ws.append(c2ws_i[0])

            cams2world = torch.stack(c2ws, dim=0).cpu() # [N, 4, 4]
            focals = to_numpy(focals)

            pts3d = to_numpy(pts3d)
            msk = to_numpy(msk)
            conf = to_numpy([x[0] for x in conf.split(1, dim=0)])
            rgbimg = to_numpy(rgbimg)

        return rgbimg, pts3d, msk, focals, cams2world, intrinsics
    

if __name__ == '__main__':
    device = "cuda"
    model_name = "MVD"
    weights_path="/home/siyanhu/Gits/mvdust3r/checkpoints/MVD.pth"
    if model_name == "MVD":
        model = AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, 1e9), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, GS = True, sh_degree=0, pts_head_config = {'skip':True})
        model.to(device)
        model_loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(get_local_path(weights_path)).to(device)
        state_dict_loaded = model_loaded.state_dict()
        model.load_state_dict(state_dict_loaded, strict=True)
    elif model_name == "MVDp":
        model = AsymmetricCroCo3DStereoMultiView(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, 1e9), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, GS = True, sh_degree=0, pts_head_config = {'skip':True}, m_ref_flag=True, n_ref = 4)
        model.to(device)
        model_loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(get_local_path(weights_path)).to(device)
        state_dict_loaded = model_loaded.state_dict()
        model.load_state_dict(state_dict_loaded, strict=True)
    else:
        raise ValueError(f"{model_name} is not supported")

    save_dir_general = '/home/siyanhu/Gits/mvdust3r/outputs'
    fio.ensure_dir(save_dir_general)

    scene_name = 'scene_corridor'
    data_dir = f'/media/siyanhu/T7/HKUST/{scene_name}/hloc_gopro/datasets'
    
    seq_dirs = fio.traverse_dir(data_dir, full_path=True, towards_sub=False)
    seq_dirs = sorted(seq_dirs, key=extract_number_foldername)
    seq = random.choice(seq_dirs)
    img_paths = fio.traverse_dir(seq, full_path=True, towards_sub=False)
    img_paths = fio.filter_ext(img_paths, filter_out_target=False, ext_set=fio.img_ext_set)
    img_paths = sorted(img_paths, key=extract_number_filename)
    
    test_loop = 10
    predictor = CSlamPredictor(model=model, device=device)
    save_dir = fio.createPath(fio.sep, [save_dir_general, scene_name], fio.get_current_timestamp(format_str="%Y%m%d"))
    start_index=0
    if fio.file_exist(save_dir):
        fio.delete_folder(save_dir)
    fio.ensure_dir(save_dir)
    for current_loop in range(test_loop):
        img_paths_subset, img_paths_subset_index = select_random_chunk(my_list=img_paths, start_index=start_index)
        predictor.generate_new_scene(file_list=img_paths_subset, save_dir=save_dir, device=device, index=start_index)
        start_index = img_paths_subset_index[-3] + 1