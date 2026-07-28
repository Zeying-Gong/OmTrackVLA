# Use cleaned agent by default, fallback to original if needed
from trained_agent import evaluate_agent
import argparse
import habitat
from habitat.datasets import make_dataset
import evt_bench
import numpy as np
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["eval", "train"],
        required=True,
        help="run type",
    )

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--split-num",
        type=int,
        default=7,
        required=False,
    )

    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on episodes after splitting; useful for visualization smoke runs.",
    )
    parser.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help="Evaluate exactly one dataset episode ID before splitting.",
    )
    parser.add_argument(
        "--episode-scene",
        type=str,
        default=None,
        help="Optional scene substring used together with --episode-id.",
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(run_type: str, exp_config: str, split_id: int, split_num: int, save_path: str, max_episodes: int, episode_id: str, episode_scene: str, opts: None) -> None:
    config = habitat.get_config(exp_config, opts)
    random.seed(config.habitat.simulator.seed)
    np.random.seed(config.habitat.simulator.seed)

    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    if episode_id is not None:
        matches = [
            ep for ep in dataset.episodes
            if str(ep.episode_id) == episode_id
            and (episode_scene is None or episode_scene in ep.scene_id)
        ]
        if not matches:
            raise ValueError(
                f"Episode ID {episode_id!r} with scene {episode_scene!r} was not found"
            )
        dataset.episodes = matches
    dataset_split = dataset.get_splits(split_num, allow_uneven_splits=True)[split_id]
    if max_episodes is not None:
        dataset_split.episodes = dataset_split.episodes[:max_episodes]

    if run_type == "eval":
        evaluate_agent(config, dataset_split, save_path)
    else:
        raise ValueError("Not supported now")
    
    return
 

if __name__ == "__main__":
    main()
