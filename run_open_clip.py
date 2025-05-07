from src.models import OpenCLIPModel
from copy import deepcopy
from utils import *
import os
from task_vectors.src.task_vectors import TaskVector
from permutations.permutation_spec import CLIP_Visual_PermutationSpecBuilder
from permutations.weights_matcher import WeightMatcher, LayerIterationOrder
from permutations.utils import apply_permutation_to_statedict
from pathlib import Path
import pickle

try:
    import wandb
    import wandbbq
    os.environ["WANDB__SERVICE_WAIT"] = "800"
except ImportError:
    wandb = None

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def parse_arguments():
    parser = argparse.ArgumentParser(description='CLIP zero-shot evaluation')
    parser.add_argument('--seed', default=33, type=int, help='Seed for initializing training.')
    parser.add_argument('--dataset', type=str, default='eurosat', choices=['cifar100', 'eurosat'], help="Dataset to evaluate.")
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size.')
    parser.add_argument('--workers', default=4, type=int, help='Number of workers for data loading.')
    parser.add_argument('--arch', default='ViT-B-16', type=str, help='Model architecture.')
    parser.add_argument('--pretraining_backbone_A', default='commonpool_l_s1b_b8k', type=str, help='Pretraining model A for backbone1.')
    parser.add_argument('--pretraining_backbone_B', default='datacomp_l_s1b_b8k', type=str, help='Pretraining model B for backbone2.')
    parser.add_argument('--finetuned_checkpoint_A', default="/work/debiasing/models/finetuned_models/eurosat/ViT-B-16/commonpool_l_s1b_b8k/best.pt", type=str, help='Path to finetuned model A.')
    parser.add_argument('--gamma', default=0.3, type=float, help='Interpolation coefficient.')
    parser.add_argument('--beta', default=5, type=float, help='Interpolation coefficient.')
    parser.add_argument('--alpha', default=0.8, type=float, help='Scaling coefficient.')
    parser.add_argument('--experiment_name', default='TMEAN(Datacomp_s to Datacomp_xl)', type=str, help='Experiment name.')
    parser.add_argument('--pre_trim', default=0., type=float, help='Experiment name.'),
    parser.add_argument('--post_trim', default=0., type=float, help='Experiment name.'),
    parser.add_argument('--max_alpha', default=1, type=float, help='Max alpha.')
    parser.add_argument('--wandb_mode',default='offline', type=str, choices=['online', 'offline', 'disabled'], help='Wandb mode')
    parser.add_argument('--base_folder', default='/leonardo_scratch/large/userexternal/frinaldi/')
    return parser.parse_args()


def main(args):

    device = setup_environment(args)
    wandbbq.init(project='Rebasin Open-Clip', entity='fillo_rinaldi-unimore', name=f'{args.arch}_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}_on_{args.dataset}', mode=args.wandb_mode, dir=args.base_folder)#online',dir='./tmp/')
    
    args.base_folder = "/work/debiasing" #fix server
    model_a, model_b, model_a_ft, preprocess = get_models(args, device)
    
    mod_openclip_a = OpenCLIPModel(model_a).clip_model
    mod_openclip_a_ft = OpenCLIPModel(model_a_ft).clip_model
    mod_openclip_b = OpenCLIPModel(model_b).clip_model
    print(f"{args.pretraining_backbone_A} to {args.pretraining_backbone_B} on {args.dataset}")
    
    t_dataloader, t_dataset, s_dataloader, s_dataset = load_dataset(args, preprocess, support=True)
    args.base_folder = "/work/debiasing/frinaldi" #fix server
    loss, acc_task_zs = evaluate_model(mod_openclip_b, t_dataloader, t_dataset, device, prompt_ensemble=True)
    print(f"Model B Original | TASK : {acc_task_zs}, loss {loss}")

    loss, acc_supp_zs = evaluate_model(mod_openclip_b, s_dataloader, s_dataset, device, prompt_ensemble=True)
    print(f"Model B Original | SUPP : {acc_supp_zs}, loss {loss}")
    wandb.log({"Model B Original | TASK": acc_task_zs, "Model B Original | SUPP": acc_supp_zs})

    ta = TaskVector(mod_openclip_a.visual, mod_openclip_a_ft.visual)
    permutation_spec_visual = CLIP_Visual_PermutationSpecBuilder(depth=mod_openclip_a.visual.transformer.layers).create_permutation_spec()


    permutations_path = Path(args.base_folder, "permutations", args.arch)

    if os.path.exists(Path(permutations_path, f'permutations_visual_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}.pkl')):
        with open(Path(permutations_path, f'permutations_visual_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}.pkl'),'rb') as f:
            permutation_visual = pickle.load(f)
        with open(Path(permutations_path, f'heads_permutation_visual_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}.pkl'), 'rb') as f:
            heads_permutation_visual = pickle.load(f)
    else:
        if not os.path.exists(permutations_path):
            os.makedirs(permutations_path)
        weight_matcher = WeightMatcher(
            ps=permutation_spec_visual,
            max_iter=100,
            p_trim=None,
            fixed=mod_openclip_b.visual.state_dict(),
            permutee=mod_openclip_a.visual.state_dict(),
            num_heads=mod_openclip_a.visual.transformer.resblocks[0].attn.num_heads,
            intra_head=True,
            layer_iteration_order=LayerIterationOrder.FORWARD)
        
        permutation_visual, heads_permutation_visual = weight_matcher.run()
    
        
        with open(Path(permutations_path, f'permutations_visual_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}.pkl'), 'wb') as f:
                pickle.dump(permutation_visual, f)
        with open(Path(permutations_path, f'heads_permutation_visual_{args.pretraining_backbone_A}_to_{args.pretraining_backbone_B}.pkl'), 'wb') as f:
                pickle.dump(heads_permutation_visual, f)
                
    #PERMUTED TASK VECTOR
    t_perm = TaskVector(vector = apply_permutation_to_statedict(permutation_spec_visual, 
                                                                permutation_visual, 
                                                                ta.vector, 
                                                                heads_permutation=heads_permutation_visual,
                                                                num_heads=mod_openclip_a.visual.transformer.resblocks[0].attn.num_heads))

    for alpha in np.linspace(0.1,args.max_alpha,9):
        log_data = {}
        model_b_t = deepcopy(mod_openclip_b)
        model_b_t.visual.load_state_dict(ta.apply_to_ptrim(
                                                    mod_openclip_b.visual, 
                                                    mod_openclip_a.visual, 
                                                    scaling_coef=alpha, 
                                                    p=args.post_trim, 
                                                    beta=args.beta,).state_dict()
        )
        loss, acc_task = evaluate_model(model_b_t, t_dataloader, t_dataset, device, prompt_ensemble=True)
        print(f"Model B + T | TASK : {acc_task}, loss {loss}")

        loss, acc_sup = evaluate_model(model_b_t, s_dataloader, s_dataset, device, prompt_ensemble=True)
        print(f"Model B + T | SUPPORT : {acc_sup}, loss {loss}")
        
        log_data.update({
                f"Model B + T TASK acc" : 100*(acc_task - acc_task_zs),
                f"Model B + T SUPPORT acc" : 100*(acc_sup - acc_supp_zs),
            })

        model_b_t = deepcopy(mod_openclip_b)
        model_b_t.visual.load_state_dict(t_perm.apply_to_ptrim(
                                                    mod_openclip_b.visual, 
                                                    mod_openclip_a.visual, 
                                                    scaling_coef=alpha, 
                                                    p=args.post_trim, 
                                                    beta=args.beta,).state_dict()
        )
        loss, acc_task = evaluate_model(model_b_t, t_dataloader, t_dataset, device, prompt_ensemble=True)
        print(f"Model B + Perm T  | TASK : {acc_task}, loss {loss}")
        
        loss, acc_sup = evaluate_model(model_b_t, s_dataloader, s_dataset, device, prompt_ensemble=True)
        print(f"Model B + Perm T | SUPPORT : {acc_sup}, loss {loss}")
        
        log_data.update({
                f"Model B + PT TASK acc" : 100*(acc_task - acc_task_zs),
                f"Model B + PT SUPPORT acc" : 100*(acc_sup - acc_supp_zs),
            })
        
        wandb.log(log_data)
    
if __name__ == '__main__':
    args = parse_arguments()
    main(args)
