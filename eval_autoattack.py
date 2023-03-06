import argparse

import torch
from tqdm import tqdm

import data_util
import model_util
import attack_util

from attack_util import ctx_noparamgrad


def parse_args():
    '''Parse input arguments'''
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eps", type=int, default=8, help="Attack budget: epsilon / 255"
    )
    parser.add_argument(
        '--norm', type=str, default='Linf', choices=['Linf', 'L2', 'L1'], help='Norm to use for attack'
    )
    parser.add_argument(
        '--data_dir', default='./data/', type=str, help="Folder to store downloaded dataset"
    )
    parser.add_argument(
        '--model_path', default='resnet_cifar10.pth', help='Filepath to the trained model'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1024, help='Batch size for attack'
    )
    parser.add_argument(
        '--log_path', type=str, default='./log_file.txt'
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    return args

def calculate_clean_and_robust_accuracy(attacker, model, dataloader, device):
    clean_correct_num = 0
    robust_correct_num = 0
    total = 0

    pbar = tqdm(total=len(dataloader))
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        batch_size = X.size(0)
        total += batch_size

        with ctx_noparamgrad(model):
            ### clean accuracy
            predictions = model(X)
            clean_correct_num += torch.sum(torch.argmax(predictions, dim = -1) == y).item()

            ### robust accuracy
            # generate perturbation
            perturbed_data = attacker.perturb(model, X, y) + X

            # predict
            predictions = model(perturbed_data)
            robust_correct_num += torch.sum(torch.argmax(predictions, dim=-1) == y).item()

        pbar.update(1)

    clean_accuracy = clean_correct_num / total
    robust_accuracy = robust_correct_num / total
    print(f"Total number of images: {total}\nClean accuracy: {clean_accuracy}\nRobust accuracy {robust_accuracy}")
    return (clean_accuracy, robust_accuracy)


def main():
    args = parse_args()
    
    # Load data
    train_loader, val_loader, test_loader, norm_layer = data_util.cifar10_dataloader(data_dir=args.data_dir)
    model = model_util.ResNet18(num_classes=10)
    model.normalize = norm_layer
    model.load(args.model_path, args.device)
    model = model.to(args.device)

    # TODO Add params from args
    att = attack_util.AT()
    nepochs = 5

    # TODO Add params from args
    pgd_attack = attack_util.PGDAttack()

    calculate_clean_and_robust_accuracy(pgd_attack, model, val_loader, args.device)

    for epoch in range(nepochs):
      loss = 0

      with tqdm(total=len(train_loader)) as pbar:
          for X, y in train_loader:
              X, y = X.to(args.device), y.to(args.device)
              loss = att.train_step(model, X, y)
              
              pbar.set_description(f"Epoch {epoch+1}/{nepochs} Loss - {round(loss, 2)}")
              pbar.update(1)

      calculate_clean_and_robust_accuracy(pgd_attack, model, val_loader, args.device)
      print(f"Finished epoch {epoch}/{nepochs}")

    ## Make sure the model is in `eval` mode.
    model.eval()
    
    # eps = args.eps / 255
    # # load attack 
    # from autoattack import AutoAttack
    # adversary = AutoAttack(model, norm=args.norm, eps=eps, log_path=args.log_path,
    #     version='standard', device=args.device)
    # 
    # l = [x for (x, y) in test_loader]
    # x_test = torch.cat(l, 0)
    # l = [y for (x, y) in test_loader]
    # y_test = torch.cat(l, 0)
    # 
    # adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)


if __name__ == "__main__":
    main()
