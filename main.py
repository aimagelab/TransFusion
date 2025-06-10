"""
Main script for running CLIP zero-shot evaluation and permutation-based transfer.
Handles argument parsing, model loading, evaluation, and permutation matching between models.
"""
# filepath: /homes/frinaldi/TransFusion/run_open_clip.py

import logging
import sys
# Configura il logging su stdout PRIMA degli import locali
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
    stream=sys.stdout
)

from copy import deepcopy
from src.models import OpenCLIPModel
from utils import *
import os
from task_vectors.src.task_vectors import TaskVector
from permutations.permutation_spec import CLIP_Visual_PermutationSpecBuilder
from permutations.weights_matcher import WeightMatcher, LayerIterationOrder
from permutations.utils import apply_permutation_to_statedict
from pathlib import Path
import pickle

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# --- Weights & Biases setup for experiment tracking ---
try:
    import wandb
    import wandbbq
    os.environ["WANDB__SERVICE_WAIT"] = "800"
except ImportError:
    wandb = None

# --- Force CUDA synchronous execution for easier debugging ---
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def main(args):
    """
    Main execution function for CLIP permutation transfer experiments.
    Loads models, datasets, runs evaluation, computes and applies permutations, and logs results.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """

    # --- Set up device and experiment tracking ---
    device = setup_environment(args)
    wandbbq.init(
        project=args.wandb_project,
        entity='fillo_rinaldi-unimore',
        name=f'{args.arch}_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}_on_{args.dataset}',
        mode=args.wandb_mode,
        dir=args.base_folder
    )

    model_a, model_b, model_a_ft, preprocess = get_models(args, device)

    # --- Wrap models for unified interface ---
    clip_a = OpenCLIPModel(model_a).clip_model
    clip_a_ft = OpenCLIPModel(model_a_ft).clip_model
    clip_b = OpenCLIPModel(model_b).clip_model
    logger.info(f"[TransFusion] Starting permutation-based transfer: {args.pretraining_backbone_A} → {args.pretraining_backbone_B} | Dataset: {args.dataset} | Architecture: {args.arch}")

    # --- Load target and support datasets (for transfer and evaluation) ---
    target_dataloader, target_dataset, support_dataloader, support_dataset = load_dataset(
        args, preprocess, support=True)

    # --- Evaluate original model B on both target and support sets ---
    loss, acc_task_zs = evaluate_model(
        clip_b, target_dataloader, target_dataset, device, prompt_ensemble=True)
    logger.info(f"[TransFusion] Baseline (Model B) | Target set: acc={acc_task_zs:.4f}, loss={loss:.4f}")

    loss, acc_supp_zs = evaluate_model(
        clip_b, support_dataloader, support_dataset, device, prompt_ensemble=True)
    logger.info(f"[TransFusion] Baseline (Model B) | Support set: acc={acc_supp_zs:.4f}, loss={loss:.4f}")
    # --- Log baseline results to wandb ---
    if wandb is not None:
        wandb.log({
            "Baseline/Target_Accuracy": acc_task_zs,
            "Baseline/Support_Accuracy": acc_supp_zs,
            "Baseline/Target_Loss": loss,
            "Baseline/Support_Loss": loss
        })
    
    ta = TaskVector(clip_a.visual, clip_a_ft.visual)
    permutation_spec_visual = CLIP_Visual_PermutationSpecBuilder(
        depth=clip_a.visual.transformer.layers).create_permutation_spec()

    # permutations_path = Path(args.base_folder, "permutations", args.arch)
    permutations_path = Path("./permutations", args.arch)

    if os.path.exists(Path(permutations_path, f'permutations_visual_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}_{args.seed}_no_bias.pkl')):
        with open(Path(permutations_path, f'permutations_visual_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}_{args.seed}_no_bias.pkl'), 'rb') as f:
            permutation_visual = pickle.load(f)
        with open(Path(permutations_path, f'heads_permutation_visual_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}_{args.seed}_no_bias.pkl'), 'rb') as f:
            heads_permutation_visual = pickle.load(f)
    else:
        if not os.path.exists(permutations_path):
            os.makedirs(permutations_path)
        weight_matcher = WeightMatcher(
            ps=permutation_spec_visual,
            max_iter=100,
            fixed=clip_b.visual.state_dict(),
            permutee=clip_a.visual.state_dict(),
            num_heads=clip_a.visual.transformer.resblocks[0].attn.num_heads,
            intra_head=True,
            layer_iteration_order=LayerIterationOrder.RANDOM)

        permutation_visual, heads_permutation_visual = weight_matcher.run()

        with open(Path(permutations_path, f'permutations_visual_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}_{args.seed}_no_bias.pkl'), 'wb') as f:
            pickle.dump(permutation_visual, f)
        with open(Path(permutations_path, f'heads_permutation_visual_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}_{args.seed}_no_bias.pkl'), 'wb') as f:
            pickle.dump(heads_permutation_visual, f)

    # PERMUTED TASK VECTOR
    t_perm = TaskVector(vector=apply_permutation_to_statedict(permutation_spec_visual,
                                                              permutation_visual,
                                                              ta.vector,
                                                              heads_permutation=heads_permutation_visual,
                                                              num_heads=clip_a.visual.transformer.resblocks[0].attn.num_heads))

    for alpha in [1]:
    # for alpha in np.linspace(, args.max_alpha, 9):
        logger.info(f"[TransFusion] Evaluating transfer with alpha={alpha}")
        log_data = {}
        model_b_t = deepcopy(clip_b)
        model_b_t.visual.load_state_dict(ta.apply_to(
            clip_b.visual,
            scaling_coef=alpha).state_dict()
        )
        loss, acc_task = evaluate_model(
            model_b_t, target_dataloader, target_dataset, device, prompt_ensemble=True)
        logger.info(f"[TransFusion] Model B + Task Vector | Target set: acc={acc_task:.4f}, loss={loss:.4f}")

        loss, acc_sup = evaluate_model(
            model_b_t, support_dataloader, support_dataset, device, prompt_ensemble=True)
        logger.info(f"[TransFusion] Model B + Task Vector | Support set: acc={acc_sup:.4f}, loss={loss:.4f}")

        log_data.update({
            "Delta/TaskVector/Target_Accuracy(%)": 100*(acc_task - acc_task_zs),
            "Delta/TaskVector/Support_Accuracy(%)": 100*(acc_sup - acc_supp_zs),
            "TaskVector/Target_Accuracy": acc_task,
            "TaskVector/Support_Accuracy": acc_sup,
            "TaskVector/Target_Loss": loss,
            "TaskVector/Support_Loss": loss
        })

        model_b_t = deepcopy(clip_b)
        model_b_t.visual.load_state_dict(t_perm.apply_to(
            clip_b.visual,
            scaling_coef=alpha).state_dict()
        )
        loss, acc_task = evaluate_model(
            model_b_t, target_dataloader, target_dataset, device, prompt_ensemble=True)
        logger.info(f"[TransFusion] Model B + Permuted Task Vector | Target set: acc={acc_task:.4f}, loss={loss:.4f}")

        loss, acc_sup = evaluate_model(
            model_b_t, support_dataloader, support_dataset, device, prompt_ensemble=True)
        logger.info(f"[TransFusion] Model B + Permuted Task Vector | Support set: acc={acc_sup:.4f}, loss={loss:.4f}")

        log_data.update({
            "Delta/PermutedTaskVector/Target_Accuracy(%)": 100*(acc_task - acc_task_zs),
            "Delta/PermutedTaskVector/Support_Accuracy(%)": 100*(acc_sup - acc_supp_zs),
            "PermutedTaskVector/Target_Accuracy": acc_task,
            "PermutedTaskVector/Support_Accuracy": acc_sup,
            "PermutedTaskVector/Target_Loss": loss,
            "PermutedTaskVector/Support_Loss": loss
        })

        if wandb is not None:
            wandb.log(log_data)


if __name__ == '__main__':
    """
    Entry point for the script. Parses arguments and runs the main function.
    """
    args = parse_arguments()
    main(args)
