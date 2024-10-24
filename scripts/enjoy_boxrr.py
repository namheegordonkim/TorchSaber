from argparse import ArgumentParser
from imgui_bundle import immapp
from imgui_bundle._imgui_bundle import imgui, hello_imgui
from pyvista_imgui import ImguiPlotter
from scipy.spatial.transform import Rotation
from torch_saber import TorchSaber
from torch_saber.utils.bsmg_xror_utils import get_xbo_np, extract_3p_with_60fps, open_bsmg_or_boxrr
from torch_saber.utils.data_utils import SegmentSampler, GameSegment, MovementSegment
from torch_saber.utils.pose_utils import expm_to_quat, unity_to_zup
from torch_saber.viz.visual_data import ObstacleVisual, NoteVisual, GenericVisual
from torch_saber.xror.xror import XROR
import numpy as np
import pyvista as pv
import torch


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class AppState:
    def __init__(
        self,
        saber_visuals: np.ndarray,
        axes_visuals: np.ndarray,
        note_visuals: np.ndarray,
        bomb_visuals: np.ndarray,
        obstacle_visuals: np.ndarray,
        note_xyzs: np.ndarray,
        note_quat: np.ndarray,
        bomb_xyzs: np.ndarray,
        bomb_quat: np.ndarray,
        obstacle_verts: np.ndarray,
        collide_yes_across_time: np.ndarray,
        good_yes_across_time: np.ndarray,
        notes: np.ndarray,
        three_p: np.ndarray,
    ):
        self.saber_visuals = saber_visuals
        self.axes_visuals = axes_visuals
        self.note_visuals = note_visuals
        self.bomb_visuals = bomb_visuals
        self.obstacle_visuals = obstacle_visuals
        self.note_xyzs = note_xyzs
        self.note_quat = note_quat
        self.bomb_xyzs = bomb_xyzs
        self.bomb_quat = bomb_quat
        self.obstacle_verts = obstacle_verts
        self.collide_yes_across_time = collide_yes_across_time
        self.good_yes_across_time = good_yes_across_time
        self.notes = notes
        self.three_p = three_p

        # GUI state parameters, in alphabetical order
        self.first_time = True
        self.show_axes = False
        self.playing = True
        self.frame = 0


def setup_and_run_gui(pl: ImguiPlotter, app_state: AppState):
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Viewer"
    runner_params.app_window_params.window_geometry.size = (1280, 720)

    def gui():
        hello_imgui.apply_theme(hello_imgui.ImGuiTheme_.imgui_colors_dark)

        viewport_size = imgui.get_window_viewport().size

        # PyVista portion
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))
        imgui.set_next_window_pos(imgui.ImVec2(viewport_size.x // 2, 0))
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin(
            "ImguiPlotter",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move,
        )
        # render the plotter's contents here
        pl.render_imgui()
        imgui.end()

        # GUI portion
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin(
            "Controls",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move,
        )

        changed, app_state.show_axes = imgui.checkbox("Show Axes", app_state.show_axes)
        changed, app_state.playing = imgui.checkbox("Playing", app_state.playing)
        changed, app_state.frame = imgui.slider_int("Frame", app_state.frame, 0, app_state.note_xyzs.shape[1])

        imgui.end()

        if app_state.playing:
            app_state.frame += 1
            if app_state.frame >= app_state.note_xyzs.shape[1]:
                app_state.playing = False

        # Saber and axes visualization
        colors = {
            0: np.array([[255.0, 0.0, 132.0]]) / 255,
            1: np.array([[0.0, 229.0, 255.0]]) / 255,
        }
        for i, xyzexpm in enumerate(app_state.three_p[0, app_state.frame]):
            xyz = xyzexpm[:3]
            quat = expm_to_quat(xyzexpm[3:])
            xyz, quat = unity_to_zup(xyz, quat)
            pos = xyz
            rot = Rotation.from_quat(quat).as_matrix()
            m = np.eye(4)
            m[:3, 3] = pos
            m[:3, :3] = rot
            # Sabers
            if 1 <= i <= 2:
                app_state.saber_visuals[i - 1].actor.user_matrix = m
                app_state.saber_visuals[i - 1].actor.SetVisibility(True)
                if app_state.collide_yes_across_time[0, app_state.frame, i - 1].any():
                    saber_color = np.array([[1.0, 0.0, 0.0]])
                    if app_state.good_yes_across_time[0, app_state.frame, i - 1].any():
                        saber_color = np.array([[0.0, 1.0, 0.0]])
                else:
                    saber_color = colors[i - 1]
                app_state.saber_visuals[i - 1].mesh.cell_data["color"] = saber_color.repeat(app_state.saber_visuals[i - 1].mesh.n_cells, 0) * 1
            # Axes
            if app_state.show_axes:
                app_state.axes_visuals[i].actor.user_matrix = m
                app_state.axes_visuals[i].actor.SetVisibility(True)
            else:
                app_state.axes_visuals[i].actor.SetVisibility(False)

        # Note, bomb, and obstacle visualization
        for i in range(app_state.note_xyzs.shape[2]):
            m = np.eye(4)
            note_xyz = app_state.note_xyzs[0, app_state.frame, i]
            if ~np.isnan(note_xyz).any():
                note_info = app_state.notes[0, app_state.frame, i]
                # Bloq color
                color = colors[int(note_info[-3].item())]
                app_state.note_visuals[i].bloq_mesh.cell_data["color"] = color.repeat(app_state.note_visuals[i].bloq_mesh.n_cells, 0) * 1

                m[:3, 3] = note_xyz
                m[:3, :3] = Rotation.from_quat(app_state.note_quat[0, app_state.frame, i]).as_matrix()
                app_state.note_visuals[i].bloq_actor.user_matrix = m
                app_state.note_visuals[i].arrow_actor.user_matrix = m
                app_state.note_visuals[i].bloq_actor.SetVisibility(True)
                app_state.note_visuals[i].arrow_actor.SetVisibility(True)
            else:
                app_state.note_visuals[i].bloq_actor.SetVisibility(False)
                app_state.note_visuals[i].arrow_actor.SetVisibility(False)

        # Render logic
        app_state.first_time = False

    runner_params.callbacks.show_gui = gui
    runner_params.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.no_default_window
    immapp.run(runner_params=runner_params)


def main(args, remaining_args):
    # Visualize rollout ygts on the left, rollout wins on the right
    pl = ImguiPlotter(window_size=(608, 608), lighting="three lights")
    pl.camera.position = (-7, 0, 7)
    pl.camera.focal_point = (3, 0, 0)
    pl.camera.up = (0, 0, 1)
    pl.set_background("#FFFFFF")
    pl.add_axes()
    plane = pv.Cube(center=(1, 0, 0), x_length=10, y_length=3, z_length=0.1)
    pl.add_mesh(plane, color="#FFFFFF")

    n_saber_visuals = 2
    n_axes_visuals = 3
    n_note_visuals = 20
    n_bomb_visuals = 20
    n_obstacle_visuals = 20

    note_visuals = np.empty(n_note_visuals, dtype=object)
    bomb_visuals = np.empty(n_bomb_visuals, dtype=object)
    obstacle_visuals = np.empty(n_obstacle_visuals, dtype=object)

    for i in range(n_note_visuals):
        note_visual = NoteVisual(pl)
        note_visuals[i] = note_visual

        note_visual = NoteVisual(pl)
        note_visual.bloq_mesh.cell_data["color"] = np.array([[0, 0, 0]]).repeat(note_visual.bloq_mesh.n_cells, 0)
        bomb_visuals[i] = note_visual

        obstacle_visual = ObstacleVisual(pl)
        obstacle_visuals[i] = obstacle_visual

    saber_visuals = np.empty(n_saber_visuals, dtype=object)
    axes_visuals = np.empty(n_axes_visuals, dtype=object)
    colors = {
        0: np.array([[255.0, 0.0, 132.0]]) / 255,
        1: np.array([[0.0, 229.0, 255.0]]) / 255,
        3: np.array([[0.0, 0.0, 0.0]]) / 255,
    }
    for i in range(n_saber_visuals):
        collider_mesh = pv.Cube(x_length=1.0, y_length=0.1, z_length=0.1)
        collider_mesh.cell_data["color"] = colors[i].repeat(collider_mesh.n_cells, 0) * 1
        collider_mesh.points += np.array([[0.5, 0, 0]])
        collider_actor = pl.add_mesh(collider_mesh, scalars="color", rgb=True, show_scalar_bar=False)
        saber_visual = GenericVisual(collider_mesh, collider_actor)
        saber_visuals[i] = saber_visual

    for i in range(n_axes_visuals):
        sphere_colors = {
            0: np.array([1.0, 0.0, 0.0]),
            1: np.array([0.0, 1.0, 0.0]),
            2: np.array([0.0, 0.0, 1.0]),
        }
        sphere_mesh = pv.Sphere(radius=0.05)
        sphere_mesh.cell_data["color"] = np.array([sphere_colors[i]]).repeat(sphere_mesh.n_cells, 0) * 255
        x_axis_mesh = pv.Arrow((0, 0, 0), (1, 0, 0), tip_radius=0.025, shaft_radius=0.01)
        x_axis_mesh.cell_data["color"] = np.array([[1, 0, 0]]).repeat(x_axis_mesh.n_cells, 0) * 255
        y_axis_mesh = pv.Arrow((0, 0, 0), (0, 1, 0), tip_radius=0.025, shaft_radius=0.01)
        y_axis_mesh.cell_data["color"] = np.array([[0, 1, 0]]).repeat(y_axis_mesh.n_cells, 0) * 255
        z_axis_mesh = pv.Arrow((0, 0, 0), (0, 0, 1), tip_radius=0.025, shaft_radius=0.01)
        z_axis_mesh.cell_data["color"] = np.array([[0, 0, 1]]).repeat(z_axis_mesh.n_cells, 0) * 255
        axes_mesh = x_axis_mesh + y_axis_mesh + z_axis_mesh
        axes_actor = pl.add_mesh(axes_mesh, scalars="color", rgb=True)

        axes_visual = GenericVisual(axes_mesh, axes_actor)
        axes_visuals[i] = axes_visual

    in_boxrr = "torch_saber/sample_data/74ec6271-61f8-4b50-a4be-21b4668fd1d8.xror"
    beatmap, song_info = open_bsmg_or_boxrr(None, in_boxrr)
    with open(in_boxrr, "rb") as f:
        file = f.read()
    xror = XROR.unpack(file)
    note_bags, bomb_bags, obstacle_bags = get_xbo_np(beatmap, song_info)
    frames_np = np.array(xror.data["frames"])
    my_3p_traj, _, timestamps = extract_3p_with_60fps(frames_np)

    length = timestamps.shape[0]
    my_3p_traj = my_3p_traj.reshape((-1, 3, 6))
    my_3p_traj = torch.as_tensor(my_3p_traj, dtype=torch.float, device=device)
    note_bags = torch.as_tensor(note_bags, dtype=torch.float, device=device)
    bomb_bags = torch.as_tensor(bomb_bags, dtype=torch.float, device=device)
    obstacle_bags = torch.as_tensor(obstacle_bags, dtype=torch.float, device=device)
    timestamps = torch.as_tensor(timestamps, dtype=torch.float, device=device)
    lengths = torch.tensor([length], dtype=torch.long, device=device)

    segment_sampler = SegmentSampler()
    game_segments, movement_segments = segment_sampler.sample(
        note_bags[None],
        bomb_bags[None],
        obstacle_bags[None],
        timestamps[None],
        my_3p_traj[None],
        lengths,
        length,
        1,
    )
    (
        collide_yes_across_time,
        color_yes_across_time,
        direction_yes_across_time,
        good_yes_across_time,
        opportunity_yes,
    ) = TorchSaber.get_collision_masks(movement_segments.three_p, game_segments.notes)
    note_verts, note_face_normals, note_quat = TorchSaber.get_note_verts_and_normals_and_quats(game_segments.notes)
    bomb_verts, bomb_face_normals, bomb_quat = TorchSaber.get_note_verts_and_normals_and_quats(game_segments.bombs)
    obstacle_verts, obstacle_face_normals = TorchSaber.get_obstacle_verts_and_normals(game_segments.obstacles)

    collide_yes_across_time = collide_yes_across_time.detach().cpu().numpy()
    good_yes_across_time = good_yes_across_time.detach().cpu().numpy()

    note_verts = note_verts.detach().cpu().numpy()
    note_quat = note_quat.detach().cpu().numpy()
    note_xyzs = note_verts.mean(-2)

    bomb_verts = bomb_verts.detach().cpu().numpy()
    bomb_quat = bomb_quat.detach().cpu().numpy()
    bomb_xyzs = bomb_verts.mean(-2)

    obstacle_verts = obstacle_verts.detach().cpu().numpy()

    notes = game_segments.notes.detach().cpu().numpy()
    three_p = movement_segments.three_p.detach().cpu().numpy()

    # Run the GUI
    app_state = AppState(saber_visuals, axes_visuals, note_visuals, bomb_visuals, obstacle_visuals, note_xyzs, note_quat, bomb_xyzs, bomb_quat, obstacle_verts, collide_yes_across_time, good_yes_across_time, notes, three_p)
    setup_and_run_gui(pl, app_state)


if __name__ == "__main__":
    parser = ArgumentParser()
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
