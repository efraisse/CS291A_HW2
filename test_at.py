"""
Tests our adversarial training pipeline to make sure we don't have any bugs
"""
import model_util
import attack_util
import data_util
import torch

# Set to "cuda:0" for accelerated testing on GPU
# Note: when running on CPU, change the first line of
# perturb() from torch.cuda.FloatTensor to torch.FloatTensor
DEVICE = "cpu"

# Ensures that the perturbation is always bounded by epsilon
def test_perturbation_clipping():
    # Set up test
    num_iterations = 1
    eps = 8/255

    # Just loading the model
    print("Loading the model...")
    train_loader, _, _, norm_layer = data_util.cifar10_dataloader(data_dir="./data/")
    model = model_util.ResNet18(num_classes=10)
    model.normalize = norm_layer
    model.load("resnet_cifar10.pth", DEVICE)
    model = model.to(DEVICE)

    # We are testing the attacker used while training and validating
    adversarial_training = attack_util.AT(eps=eps, model=model, device=DEVICE)
    validation_attacker = attack_util.PGDAttack(eps=eps, device=DEVICE)


    # Run a few train steps and check that the attack is clipped.
    # Note that in this test, PGD is run a few times.
    print(f"Running {num_iterations} train steps...")
    for i, (X, y) in enumerate(train_loader):
        if not (i < num_iterations):
            break

        X, y = X.to(DEVICE), y.to(DEVICE)

        print("Calculating perturbation used for adversarial training...")
        # Calculate perturbation used for adversarial training
        perturbation_used_in_training = (
            adversarial_training._pgd_attack.perturb(model, X, y))

        print("Having model take a step...")
        # Have the model take a step, to simulate learning
        adversarial_training.train_step(model, X, y)

        print("Calculating perturbation from validation")
        # Also make sure that the delta used for evaluation has
        # constrained norm
        perturbation_used_in_validation = validation_attacker.perturb(model, X, y)

        # Check the infinity norms of the perturbations
        assert((torch.max(perturbation_used_in_training) <= eps and
            torch.min(perturbation_used_in_training) >= -eps))
        assert((torch.max(perturbation_used_in_validation) <= eps and
            torch.min(perturbation_used_in_validation) >= -eps))

        print(f"Finished {i+1}/{num_iterations}!")
