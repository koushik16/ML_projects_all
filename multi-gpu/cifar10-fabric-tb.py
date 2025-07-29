import argparse
from pathlib import Path

from accelerate import optimizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
import torchvision.datasets as datasets

from einops.layers.torch import Reduce

from torchmetrics.classification import Accuracy
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import TensorBoardLogger


parser = argparse.ArgumentParser(description="multi-gpu image classification")

# arguments for fabric
parser.add_argument(
    "--accelerator",
    type=str,
    default="cpu",
    metavar="S",
    help="type of device (default: cpu)",
)
parser.add_argument(
    "--devices",
    type=int,
    default=1,
    metavar="N",
    help="number of devices per node (default: 1)",
)
parser.add_argument(
    "--num-nodes", type=int, default=1, metavar="N", help="number of nodes (default: 1)"
)
parser.add_argument(
    "--strategy",
    type=str,
    default="ddp",
    metavar="S",
    help="parallelization strategy (default: ddp)",
)
parser.add_argument(
    "--precision",
    type=str,
    default="bf16-mixed",
    metavar="S",
    help="float precision (default: bf16-mixed)",
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)

# arguments for training process
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=1,
    metavar="N",
    help="number of workers to load data (default: 1)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=14,
    metavar="N",
    help="number of epochs to train (default: 14)",
)
parser.add_argument(
    "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.7,
    metavar="M",
    help="learning rate step gamma (default: 0.7)",
)

# arguments for checkpointing
parser.add_argument(
    "--save-checkpoint-interval",
    type=int,
    default=2,
    metavar="N",
    help="number of epochs to save model after (default: 10)",
)
parser.add_argument(
    "--resume-from-checkpoint",
    type=str,
    default=None,
    metavar="S",
    help="path to checkpoint to resume training from",
)

# other arguments
parser.add_argument(
    "--dry-run", action="store_true", default=False, help="quickly check a single pass"
)

args = parser.parse_args()

# fabric setup
tb_logger = TensorBoardLogger(root_dir="./logs/tb", name="cifar10-multi-gpu") # log losses and accuracies to plot later
fabric = Fabric(
    accelerator=args.accelerator,
    devices=args.devices,
    num_nodes=args.num_nodes,
    strategy=args.strategy,
    precision=args.precision,
    loggers=[tb_logger],
)
fabric.launch()
seed_everything(args.seed)  # this step is not strictly necessary, it's just for reproducibility

# data
with fabric.rank_zero_first(local=False):  # this ensures that this is done only once
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    download_data = not Path(
        "./data/cifar-10-batches-py/"
    ).exists()  # checks if CIFAR10 already exists

    train_data = datasets.CIFAR10(
        root="./data", train=True, download=download_data, transform=transform
    )
    test_data = datasets.CIFAR10(
        root="./data", train=False, download=download_data, transform=transform
    )

train_loader = DataLoader(
    train_data,
    args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)
test_loader = DataLoader(
    test_data, args.batch_size, num_workers=args.num_workers, pin_memory=True
)

train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

# model
num_classes = 10
with fabric.init_module():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        Reduce("b c (h 2) (w 2) -> b (c h w)", "max"),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)

model, optimizer = fabric.setup(model, optimizer)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

# resuming from previous training
start_epoch = 0
if args.resume_from_checkpoint:
    fabric.print(f"Resuming from {args.resume_from_checkpoint}")
    checkpoint = fabric.load(Path("checkpoints") / args.resume_from_checkpoint)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]

# metrics
train_acc_fn = Accuracy(task="multiclass", num_classes=num_classes).to(fabric.device)
test_acc_fn = Accuracy(task="multiclass", num_classes=num_classes).to(fabric.device)

# training
for epoch in range(start_epoch, args.epoch):
    model.train()
    train_loss = 0.
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        fabric.backward(loss)
        optimizer.step()

        train_loss += loss.item()
        train_acc_fn(outputs, labels)
    scheduler.step()
    train_loss = fabric.all_gather(train_loss).sum() / len(train_loader.dataset)

    model.eval()
    test_loss = 0.
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            test_loss += loss_fn(outputs, labels).item()
            test_acc_fn(outputs, labels)
    test_loss = fabric.all_gather(test_loss).sum() / len(test_loader.dataset)


    train_acc, test_acc = train_acc_fn.compute(), test_acc_fn.compute()
    fabric.print(f"epoch: {epoch}", end=" ")
    fabric.print(f"train loss: {train_loss:.3e} train acc: {100 * train_acc:.0f}%", end=" ")
    fabric.print(f"test loss: {test_loss:.3e} test acc: {100 * test_acc:.0f}%")

    metrics = {
        "train_loss": train_loss,
        "train_acc": 100 * train_acc,
        "test_loss": test_loss,
        "test_acc": 100 * test_acc
    }
    fabric.log_dict(metrics, epoch) # the log files will be saved in ./logs/tb/cifar10-multi-gpu, you can view the plots with tensorboard

    # save checkpoint
    if epoch % args.save_checkpoint_interval == 0:
        state = {"model": model, "optimizer": optimizer, "epoch": epoch}
        fabric.save(Path("checkpoints") / f"checkpoint_{epoch}.ckpt", state)

    # reset metrics
    train_acc_fn.reset()
    test_acc_fn.reset()

    if args.dry_run:
        break
