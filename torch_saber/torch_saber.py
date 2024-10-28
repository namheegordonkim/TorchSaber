from functools import reduce

import torch
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from torch_saber.utils.pose_utils import unity_to_zup, quat_rotate, expm_to_quat_torch


class TorchSaber:

    @staticmethod
    def evaluate(my_3p_traj: torch.Tensor, note_bags: torch.Tensor, batch_size: int = None):
        """
        Given a batch of 3p trajecotries (normalized to match PHC's height) and the sequence of note bags,
        Evaluate the f1 score of the trajectory, assuming the dimensions of the hitboxes.
        Many simplifying assumptions are made here, but as a baseline this should be fine.

        my_3p_traj has shape (B, T, 3, 6)
        where the 6-dim feature is xyz and expm
        note_bags has shape (B, T, 20, 5)
        where the 5-dim feature is time, x, y, color, angle
        """
        idxs = torch.arange(my_3p_traj.shape[1], device="cuda")
        batch_idxs = torch.split(idxs, batch_size if batch_size is not None else my_3p_traj.shape[1])
        batch_reses = []
        for batch_i in batch_idxs:
            res = TorchSaber.get_collision_masks(my_3p_traj[:, batch_i], note_bags[:, batch_i])
            batch_reses.append(res)
        (
            collide_yes_across_time,
            color_yes_across_time,
            direction_yes_across_time,
            good_yes_across_time,
            opportunity_yes,
        ) = list(reduce(lambda acc, res: [torch.cat([a, r], dim=1) for a, r in zip(acc, res)], batch_reses))

        n_opportunities = torch.sum(opportunity_yes, dim=(1, 2))
        n_hits = torch.sum(torch.any(collide_yes_across_time, dim=2), dim=(1, 2))
        n_misses = n_opportunities - n_hits
        n_goods = torch.sum(good_yes_across_time, dim=(1, 2, 3))
        f1 = (2 * n_goods) / (2 * n_goods + (n_hits - n_goods) + n_misses)
        f1[f1.isnan()] = 0

        return f1, n_hits, n_misses, n_goods

    @staticmethod
    def get_note_verts_and_normals_and_quats(note_bags: torch.Tensor):
        """
        Get the vertices and face normals of notes in the bag post transform
        """
        note_collider_mesh = pv.Cube(x_length=0.4, y_length=0.4, z_length=0.4)
        note_collider_verts = note_collider_mesh.points
        note_verts = note_collider_verts[None, None, None].repeat(note_bags.shape[0], 0).repeat(note_bags.shape[1], 1).repeat(20, 2)
        note_verts = torch.tensor(note_verts, dtype=torch.float, device="cuda")
        note_angle_degrees = np.array([0, 180, -90, 90, -45, 45, -135, 135, 0])
        tmp = note_bags.detach().cpu().numpy() * 1
        tmp = np.where(np.isnan(tmp), 0, tmp)
        tmp = tmp.astype(int)
        note_angles = np.where(
            np.isnan(note_bags[:, :, :, -2].detach().cpu().numpy()),
            np.nan,
            note_angle_degrees[tmp[:, :, :, -2]],
        )
        note_quat = Rotation.from_euler("x", note_angles.reshape(-1), degrees=True).as_quat().reshape((note_angles.shape[0], note_angles.shape[1], 20, 4))
        note_quat = torch.tensor(note_quat, dtype=torch.float, device="cuda")
        note_verts = quat_rotate(
            note_quat[:, :, :, None].repeat_interleave(8, 3),
            note_verts,
        )

        plane_width = 1.0
        plane_height = 0.5
        plane_left = 0 - plane_width / 2
        plane_bottom = 0 - plane_height / 2
        plane_right = 0 + plane_width / 2
        plane_top = 0 + plane_height / 2
        plane_grid = np.array(
            np.meshgrid(
                np.linspace(plane_right, plane_left, 4),  # note: left and right are flipped for unity
                np.linspace(plane_bottom, plane_top, 3),
            )
        ).transpose((2, 1, 0))
        plane_grid = torch.tensor(plane_grid, dtype=torch.float, device="cuda")

        note_pos = plane_grid[tmp[..., 3], tmp[..., 4]]
        note_pos = torch.concatenate([note_bags[..., [0]] * 10, note_pos], dim=-1)
        note_pos[..., 2] += 1.5044 - 0.333
        note_verts += note_pos[:, :, :, None]

        note_collider_normals = note_collider_mesh.face_normals
        note_face_normals = note_collider_normals[None, None, None].repeat(note_bags.shape[0], 0).repeat(note_bags.shape[1], 1).repeat(20, 2)
        note_face_normals = torch.tensor(note_face_normals, dtype=torch.float, device="cuda")
        note_face_normals = quat_rotate(
            note_quat[:, :, :, None].repeat_interleave(6, 3),
            note_face_normals,
        )

        return note_verts, note_face_normals, note_quat

    @staticmethod
    def get_obstacle_verts_and_normals(obstacle_bags: torch.Tensor):
        """
        Get the vertices and face normals of notes in the bag post transform
        """
        obstacle_collider_mesh = pv.Cube(x_length=0.4, y_length=0.4, z_length=0.4)
        obstacle_collider_verts = obstacle_collider_mesh.points
        obstacle_verts = obstacle_collider_verts[None, None, None].repeat(obstacle_bags.shape[0], 0).repeat(obstacle_bags.shape[1], 1).repeat(20, 2)
        obstacle_verts = torch.tensor(obstacle_verts, dtype=torch.float, device="cuda")
        tmp = obstacle_bags.detach().cpu().numpy() * 1
        tmp = np.where(np.isnan(tmp), 0, tmp)
        tmp = tmp.astype(int)

        plane_width = 1.0
        plane_height = 0.5
        plane_left = 0 - plane_width / 2
        plane_bottom = 0 - plane_height / 2
        plane_right = 0 + plane_width / 2
        plane_top = 0 + plane_height / 2

        plane_leftright = np.linspace(plane_right, plane_left, 4)  # note: left and right are flipped for unity
        plane_bottomtop = np.linspace(plane_bottom, plane_top, 3)
        plane_width_interval = plane_leftright[1] - plane_leftright[0]
        plane_height_interval = plane_bottomtop[1] - plane_bottomtop[0]

        plane_grid = np.array(
            np.meshgrid(
                plane_leftright,
                plane_bottomtop,
            )
        ).transpose((2, 1, 0))
        plane_grid = torch.tensor(plane_grid, dtype=torch.float, device="cuda")

        # Locate base to the correct place based on x and y
        obstacle_pos = plane_grid[tmp[..., 5], tmp[..., 6]]
        obstacle_pos = torch.concatenate([obstacle_bags[..., [0]] * 10, obstacle_pos], dim=-1)
        obstacle_pos[..., 2] += 1.5044 - 0.333
        obstacle_verts += obstacle_pos[:, :, :, None]

        # Depth based on duration (index 4)
        obstacle_verts[..., -4:, 0] += obstacle_bags[..., [4]] * 10
        # Top height based on index -1
        # If height is 1 then no offset added. If height is 2, then add 1 * height_interval
        obstacle_verts[..., [1, 2, 6, 7], 2] += (obstacle_bags[..., [-1]] - 1) * plane_height_interval

        # left-right based on index -2
        # If width is 1 then no offset added. If width is 2, then add 1 * width_interval
        # left vertices
        obstacle_verts[..., [0, 1, 4, 7], 1] += (obstacle_bags[..., [-2]] - 1) * np.clip(plane_width_interval, a_min=-np.inf, a_max=0)
        # right vertices
        obstacle_verts[..., [2, 3, 5, 6], 1] += (obstacle_bags[..., [-2]] - 1) * np.clip(plane_width_interval, a_min=0, a_max=np.inf)

        obstacle_verts[obstacle_verts.isnan().any(-1)] = torch.nan

        obstacle_face_normals = obstacle_collider_mesh.face_normals
        obstacle_face_normals = obstacle_face_normals[None, None, None].repeat(obstacle_bags.shape[0], 0).repeat(obstacle_bags.shape[1], 1).repeat(20, 2)
        obstacle_face_normals = torch.tensor(obstacle_face_normals, dtype=torch.float, device="cuda")
        obstacle_face_normals[obstacle_verts.isnan().any(-1).any(-1)] = torch.nan

        return obstacle_verts, obstacle_face_normals

    @staticmethod
    def box_box_collision_from_verts_and_normals(verts1: torch.Tensor, normals1: torch.Tensor, verts2: torch.Tensor, normals2: torch.Tensor):
        edge_idxs = pv.Cube().regular_faces.reshape(-1, 2)
        edges1 = verts1[..., edge_idxs[:, 0], :] - verts1[..., edge_idxs[:, 1], :]
        edges2 = verts2[..., edge_idxs[:, 0], :] - verts2[..., edge_idxs[:, 1], :]
        edge_crosses = (
            torch.cross(
                edges1[:, :, :, :, None, None],
                edges2[:, :, None, None],
                dim=-1,
            )
            .permute((0, 1, 2, 4, 3, 5, 6))
            .contiguous()
        )
        edge_crosses = edge_crosses.view(
            (
                edge_crosses.shape[0],
                edge_crosses.shape[1],
                edge_crosses.shape[2],
                edge_crosses.shape[3],
                -1,
                3,
            )
        )
        # The end-all-be-all for collision precompute
        all_cand_axes_across_time = torch.concatenate(
            [
                edge_crosses,
                normals1[:, :, :, None].repeat_interleave(edge_crosses.shape[3], 3),
                normals2[:, :, None].repeat_interleave(edge_crosses.shape[2], 2),
            ],
            dim=-2,
        )
        all_cand_axes_across_time /= (torch.norm(all_cand_axes_across_time, dim=-1, keepdim=True) + 1e-10)

        proj1 = torch.sum(
            verts1[:, :, :, None, None] * all_cand_axes_across_time[:, :, :, :, :, None],
            dim=-1,
        )
        proj2 = torch.sum(
            verts2[:, :, None, :, None] * all_cand_axes_across_time[:, :, :, :, :, None],
            dim=-1,
        )
        min1 = proj1.min(-1)[0]
        max1 = proj1.max(-1)[0]
        min2 = proj2.min(-1)[0]
        max2 = proj2.max(-1)[0]
        collide_yes = torch.all(max1 >= min2, dim=-1) & torch.all(max2 >= min1, dim=-1)
        return collide_yes

    @staticmethod
    def get_collision_masks(my_3p_traj: torch.Tensor, note_bags: torch.Tensor, obstacle_bags: torch.Tensor):
        saber_collider_verts_across_time, saber_collider_normals_across_time, saber_quat = TorchSaber.get_saber_verts_and_normals_and_quats(my_3p_traj)
        note_collider_verts_across_time, note_collider_normals_across_time, note_quat = TorchSaber.get_note_verts_and_normals_and_quats(note_bags)
        three_p_verts_across_time, three_p_normals_across_time = TorchSaber.get_3p_verts_and_normals(my_3p_traj)
        obstacle_verts_across_time, obstacle_normals_across_time = TorchSaber.get_obstacle_verts_and_normals(obstacle_bags)

        tmp = note_bags.detach().cpu().numpy() * 1
        tmp = np.where(np.isnan(tmp), 0, tmp)
        tmp = tmp.astype(int)

        saber_box_collision_yeses = TorchSaber.box_box_collision_from_verts_and_normals(saber_collider_verts_across_time, saber_collider_normals_across_time, note_collider_verts_across_time, note_collider_normals_across_time)
        three_p_obstacle_collision_yeses = TorchSaber.box_box_collision_from_verts_and_normals(three_p_verts_across_time, three_p_normals_across_time, obstacle_verts_across_time, obstacle_normals_across_time)

        color_onehots = torch.eye(2, dtype=torch.bool, device="cuda")[tmp[:, :, :, -3]].permute((0, 1, 3, 2))
        color_yes_across_time = saber_box_collision_yeses & color_onehots

        offsets = torch.tensor([[[[1, 0, 0]]]], dtype=torch.float, device="cuda").repeat_interleave(note_bags.shape[0], 0).repeat_interleave(note_bags.shape[1], 1).repeat_interleave(2, 2)
        offsets = quat_rotate(saber_quat.contiguous(), offsets)
        offset_vels = offsets * 0
        offset_vels[:, 1:] = offsets[:, 1:] - offsets[:, :-1]
        offset_vels /= 1 / 60

        cut_dir_vecs_across_time = torch.tensor([0, 0, 1], dtype=torch.float, device="cuda")[None, None, None].repeat_interleave(note_bags.shape[0], 0).repeat_interleave(note_bags.shape[1], 1).repeat_interleave(20, 2)
        cut_dir_vecs_across_time = quat_rotate(note_quat, cut_dir_vecs_across_time)
        dots_across_time = torch.sum(cut_dir_vecs_across_time[:, :, None] * offset_vels[:, :, :, None], dim=-1)
        direction_yes_across_time = dots_across_time > 0.2
        good_yes_across_time = saber_box_collision_yeses & color_yes_across_time & direction_yes_across_time
        note_pos = note_collider_verts_across_time.mean(-2)
        opportunity_yes = note_pos[..., 0] < 1

        return (
            saber_box_collision_yeses,
            color_yes_across_time,
            direction_yes_across_time,
            good_yes_across_time,
            opportunity_yes,
            three_p_obstacle_collision_yeses,
        )

    @staticmethod
    def get_saber_verts_and_normals_and_quats(my_3p_traj: torch.Tensor):
        my_3p_xyz, my_3p_expm = (
            my_3p_traj[..., :3] * 1,
            my_3p_traj[..., 3:] * 1,
        )
        my_3p_quat = expm_to_quat_torch(my_3p_expm)
        my_3p_xyz, my_3p_quat = unity_to_zup(my_3p_xyz, my_3p_quat)

        # For getting tip velocity, just precompute all offsets
        offsets = torch.tensor([[[[1, 0, 0]]]], dtype=torch.float, device="cuda").repeat_interleave(my_3p_traj.shape[0], 0).repeat_interleave(my_3p_traj.shape[1], 1).repeat_interleave(3, 2)
        offsets = quat_rotate(my_3p_quat, offsets)

        offset_vels = offsets * 0
        offset_vels[:, 1:] = offsets[:, 1:] - offsets[:, :-1]
        offset_vels /= 1 / 60

        # Pre-compute collision stuff using vertices
        saber_collider_mesh = pv.Cube(x_length=1.0, y_length=0.1, z_length=0.1)
        saber_collider_verts = saber_collider_mesh.points
        saber_collider_verts_across_time = saber_collider_verts[None, None, None].repeat(my_3p_traj.shape[0], 0).repeat(my_3p_traj.shape[1], 1).repeat(2, 2)
        saber_collider_verts_across_time = torch.tensor(saber_collider_verts_across_time, dtype=torch.float, device="cuda")
        saber_quats = my_3p_quat[:, :, 1:]
        saber_xyzs = my_3p_xyz[:, :, 1:]

        saber_collider_verts_across_time += torch.tensor([0.5, 0, 0], dtype=torch.float, device="cuda")[None, None, None, None]
        saber_collider_verts_across_time = quat_rotate(
            saber_quats[:, :, :, None].repeat_interleave(8, 3),
            saber_collider_verts_across_time,
        )
        saber_collider_verts_across_time += saber_xyzs[:, :, :, None]

        saber_collider_normals = saber_collider_mesh.face_normals
        saber_collider_normals_across_time = saber_collider_normals[None, None, None].repeat(my_3p_traj.shape[0], 0).repeat(my_3p_traj.shape[1], 1).repeat(2, 2)
        saber_collider_normals_across_time = torch.tensor(saber_collider_normals_across_time, dtype=torch.float, device="cuda")
        saber_collider_normals_across_time = quat_rotate(
            saber_quats[:, :, :, None].repeat_interleave(6, 3),
            saber_collider_normals_across_time,
        )
        return (
            saber_collider_verts_across_time,
            saber_collider_normals_across_time,
            saber_quats,
        )

    @staticmethod
    def get_3p_verts_and_normals(my_3p_traj: torch.Tensor):
        collider_mesh = pv.Cube(x_length=0.1, y_length=0.1, z_length=0.1)
        collider_verts = collider_mesh.points
        collider_verts_across_time = collider_verts[None, None, None].repeat(my_3p_traj.shape[0], 0).repeat(my_3p_traj.shape[1], 1).repeat(3, 2)
        collider_verts_across_time = torch.tensor(collider_verts_across_time, dtype=torch.float, device="cuda")
        my_3p_xyz, my_3p_expm = (
            my_3p_traj[..., :3] * 1,
            my_3p_traj[..., 3:] * 1,
        )
        my_3p_quat = expm_to_quat_torch(my_3p_expm)
        my_3p_xyz, my_3p_quat = unity_to_zup(my_3p_xyz, my_3p_quat)
        collider_verts_across_time += my_3p_xyz[:, :, :, None]
        collider_normals = collider_mesh.face_normals
        collider_normals_across_time = collider_normals[None, None, None].repeat(my_3p_traj.shape[0], 0).repeat(my_3p_traj.shape[1], 1).repeat(3, 2)
        collider_normals_across_time = torch.tensor(collider_normals_across_time, dtype=torch.float, device="cuda")

        return collider_verts_across_time, collider_normals_across_time
