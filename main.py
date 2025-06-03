"""
Main script for running CLIP zero-shot evaluation and permutation-based transfer.
Handles argument parsing, model loading, evaluation, and permutation matching between models.
"""
# filepath: /homes/frinaldi/TransFusion/run_open_clip.py
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
    print(f"{args.pretraining_backbone_A} to {args.pretraining_backbone_B} on {args.dataset}")

    # --- Load target and support datasets (for transfer and evaluation) ---
    target_dataloader, target_dataset, support_dataloader, support_dataset = load_dataset(
        args, preprocess, support=True)

    # --- Evaluate original model B on both target and support sets ---
    loss, acc_task_zs = evaluate_model(
        clip_b, target_dataloader, target_dataset, device, prompt_ensemble=True)
    print(f"Model B Original | TASK : {acc_task_zs}, loss {loss}")

    loss, acc_supp_zs = evaluate_model(
        clip_b, support_dataloader, support_dataset, device, prompt_ensemble=True)
    print(f"Model B Original | SUPP : {acc_supp_zs}, loss {loss}")
    # --- Log baseline results to wandb ---
    wandb.log({
        "Model B Original | TASK": acc_task_zs,
        "Model B Original | SUPP": acc_supp_zs
    })
    
    ta = TaskVector(clip_a.visual, clip_a_ft.visual)
    permutation_spec_visual = CLIP_Visual_PermutationSpecBuilder(
        depth=clip_a.visual.transformer.layers).create_permutation_spec()

    permutations_path = Path(args.base_folder, "permutations", args.arch)

    if os.path.exists(Path(permutations_path, f'permutations_visual_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}.pkl')):
        with open(Path(permutations_path, f'permutations_visual_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}.pkl'), 'rb') as f:
            permutation_visual = pickle.load(f)
        with open(Path(permutations_path, f'heads_permutation_visual_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}.pkl'), 'rb') as f:
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
            layer_iteration_order=LayerIterationOrder.FORWARD)

        permutation_visual, heads_permutation_visual = weight_matcher.run()

        with open(Path(permutations_path, f'permutations_visual_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}.pkl'), 'wb') as f:
            pickle.dump(permutation_visual, f)
        with open(Path(permutations_path, f'heads_permutation_visual_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}.pkl'), 'wb') as f:
            pickle.dump(heads_permutation_visual, f)

    # PERMUTED TASK VECTOR
    t_perm = TaskVector(vector=apply_permutation_to_statedict(permutation_spec_visual,
                                                              permutation_visual,
                                                              ta.vector,
                                                              heads_permutation=heads_permutation_visual,
                                                              num_heads=clip_a.visual.transformer.resblocks[0].attn.num_heads))

    for alpha in np.linspace(0.1, args.max_alpha, 9):
        log_data = {}
        model_b_t = deepcopy(clip_b)
        model_b_t.visual.load_state_dict(ta.apply_to(
            clip_b.visual,
            clip_a.visual,
            scaling_coef=alpha).state_dict()
        )
        loss, acc_task = evaluate_model(
            model_b_t, target_dataloader, target_dataset, device, prompt_ensemble=True)
        print(f"Model B + T | TASK : {acc_task}, loss {loss}")

        loss, acc_sup = evaluate_model(
            model_b_t, support_dataloader, support_dataset, device, prompt_ensemble=True)
        print(f"Model B + T | SUPPORT : {acc_sup}, loss {loss}")

        log_data.update({
            f"Model B + T TASK acc": 100*(acc_task - acc_task_zs),
            f"Model B + T SUPPORT acc": 100*(acc_sup - acc_supp_zs),
        })

        model_b_t = deepcopy(clip_b)
        model_b_t.visual.load_state_dict(t_perm.apply_to(
            clip_b.visual,
            clip_a.visual,
            scaling_coef=alpha).state_dict()
        )
        loss, acc_task = evaluate_model(
            model_b_t, target_dataloader, target_dataset, device, prompt_ensemble=True)
        print(f"Model B + Perm T  | TASK : {acc_task}, loss {loss}")

        loss, acc_sup = evaluate_model(
            model_b_t, support_dataloader, support_dataset, device, prompt_ensemble=True)
        print(f"Model B + Perm T | SUPPORT : {acc_sup}, loss {loss}")

        log_data.update({
            f"Model B + PT TASK acc": 100*(acc_task - acc_task_zs),
            f"Model B + PT SUPPORT acc": 100*(acc_sup - acc_supp_zs),
        })

        wandb.log(log_data)


if __name__ == '__main__':
    """
    Entry point for the script. Parses arguments and runs the main function.
    """
    args = parse_arguments()
    main(args)
