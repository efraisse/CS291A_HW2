import data_util
import argparse

import torch
from tqdm import tqdm

import model_util
import attack_util
from autoattack import AutoAttack

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
        '--model_path', default='models/resnet_cifar10.pth', help='Filepath to the trained model'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128, help='Batch size for attack'
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
    # redeclare the semisup_train_loader and semisup_ogdata_mix inside the epochs to reshuffle the data
    semisup_train_loader = data_util.ti500k_dataloader(batch_size = args.batch_size)
    train_loader, val_loader, test_loader, norm_layer = data_util.cifar10_dataloader(batch_size = args.batch_size, data_dir=args.data_dir)
    semisup_ogdata_mix = data_util.ti500k_and_ogdata_dataloader(batch_size = args.batch_size)
    model = model_util.ResNet18(num_classes=10)
    model.normalize = norm_layer
    model.load(args.model_path, args.device)
    model = model.to(args.device)
    
    # model.eval()

    # TODO Add params from args
    att = attack_util.AT(model = model, device = args.device)
    nepochs = 50
    
    max_robust = 0

    # TODO Add params from args
    # PGD
    pgd_attack = attack_util.PGDAttack(device = args.device)
    # pgd_attack = attack_util.PGDAttack(device = args.device, alpha=args.eps, attack_step = 1)

    calculate_clean_and_robust_accuracy(pgd_attack, att.model, val_loader, args.device)
    
    # going to halve the linear schedule so instead of 75/90/100 will do 30/45/50
    # for the 4684 model, will start with learning rate 0.01 for 30 epochs and then again at 45
    # for the 4684 PGD50 model, will start with regular learning rate for 10 epochs, then drop to 20, then 30?
    # model_name = "TRADES_OGPARAMS_semisup_ogdata_mix.pth"
    # model_name = "TRADES_OGPARAMS_semisup_ogdata_mix_4684model.pth"
    model_name = "TRADES_OGPARAMS_semisup_ogdata_mix_4684model_NOVAL.pth"

    for epoch in range(nepochs):
      loss = 0
      
      semisup_ogdata_mix = data_util.ti500k_and_ogdata_dataloader(batch_size = args.batch_size)

      with tqdm(total=len(semisup_ogdata_mix)) as pbar:
          for X, y in semisup_ogdata_mix:

    #   with tqdm(total=len(train_loader)) as pbar:
    #       for X, y in train_loader:
          
    #   with tqdm(total=len(semisup_train_loader)) as pbar:
    #       for X, y in semisup_train_loader:
              X, y = X.to(args.device), y.to(args.device)
              loss = att.train_step(model, X, y)
              
              pbar.set_description(f"Epoch {epoch+1}/{nepochs} Loss - {round(loss, 2)}")
              pbar.update(1)

      _, robust_accuracy = calculate_clean_and_robust_accuracy(pgd_attack, att.model, val_loader, args.device)
          
      if robust_accuracy > max_robust:  
        att.model.save(model_name)
        max_robust = robust_accuracy
          
      print(f"Finished epoch {epoch + 1}/{nepochs}")
      att.schedule.step()

    if robust_accuracy > max_robust:  
        att.model.save(model_name)
        max_robust = robust_accuracy
    
    ## Make sure the model is in `eval` mode.
    att.model.eval()

    pgd_attack = attack_util.PGDAttack(attack_step = 50, device = args.device, loss_type = "ce")
        
    calculate_clean_and_robust_accuracy(pgd_attack, att.model, test_loader, args.device)
    
    # part 2 of the assignment
    # eps = args.eps / 255
    # # load attack 
    
    # adversary = AutoAttack(model, norm=args.norm, eps=eps, log_path=args.log_path,
    #     version='standard', device=args.device)
    
    # l = [x for (x, y) in test_loader]
    # x_test = torch.cat(l, 0)
    # l = [y for (x, y) in test_loader]
    # y_test = torch.cat(l, 0)
    
    # adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)


if __name__ == "__main__":
    main()
