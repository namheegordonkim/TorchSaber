from argparse import ArgumentParser
from torch_saber import TorchSaber
from torch_saber.utils.bsmg_xror_utils import get_xbo_np, extract_3p_with_60fps, open_bsmg_or_boxrr
from torch_saber.utils.data_utils import SegmentSampler
from torch_saber.xror.xror import XROR
import numpy as np
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main(args, remaining_args):
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

    f1, n_hits, n_misses, n_goods = TorchSaber.evaluate(movement_segments.three_p, game_segments.notes, args.batch_size)
    for i in range(f1.shape[0]):
        print(f"Evaluation for input BOXRR {i}")
        print(f"{f1[i]=:.2f}, {n_hits[i]=}, {n_misses[i]=}, {n_goods[i]=}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    args, remaining_args = parser.parse_known_args()
    main(args, remaining_args)
