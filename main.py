import json
import os
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader

from dataloader.dataset import MedicalDataSets
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize

from utils.util import AverageMeter
import utils.losses as losses
from utils.metrics import iou_score

from network.CMUNeXt import cmunext, cmunext_s, cmunext_l


def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_torch(41)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="CMUNeXt",
    choices=["CMUNeXt", "CMUNeXt-S", "CMUNeXt-L"],
    help="model",
)
parser.add_argument("--base_dir", type=str, default="./data/busi", help="dir")
parser.add_argument("--train_file_dir", type=str, default="busi_train.txt", help="dir")
parser.add_argument("--val_file_dir", type=str, default="busi_val.txt", help="dir")
parser.add_argument(
    "--base_lr", type=float, default=0.01, help="segmentation network learning rate"
)
parser.add_argument("--batch_size", type=int, default=8, help="batch_size per gpu")
parser.add_argument(
    "-c",
    "--continue_train",
    action="store_true",
    help="continue training from the last saved checkpoint (if available)",
)
args = parser.parse_args()


def getDataloader():
    img_size = 256
    train_transform = Compose(
        [
            RandomRotate90(),
            transforms.Flip(),
            Resize(img_size, img_size),
            transforms.Normalize(),
        ]
    )

    val_transform = Compose(
        [
            Resize(img_size, img_size),
            transforms.Normalize(),
        ]
    )
    db_train = MedicalDataSets(
        base_dir=args.base_dir,
        split="train",
        transform=train_transform,
        train_file_dir=args.train_file_dir,
        val_file_dir=args.val_file_dir,
    )
    db_val = MedicalDataSets(
        base_dir=args.base_dir,
        split="val",
        transform=val_transform,
        train_file_dir=args.train_file_dir,
        val_file_dir=args.val_file_dir,
    )
    print("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    trainloader = DataLoader(
        db_train, batch_size=8, shuffle=True, num_workers=8, pin_memory=False
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    return trainloader, valloader


def get_model(args):
    if args.model == "CMUNeXt":
        model = cmunext()
    elif args.model == "CMUNeXt-S":
        model = cmunext_s()
    elif args.model == "CMUNeXt-L":
        model = cmunext_l()
    else:
        model = None
        print("model err")
        exit(0)
    return model.cuda()


def train(args):
    base_lr = args.base_lr
    trainloader, valloader = getDataloader()
    model = get_model(args)
    print(
        "train file dir:{} val file dir:{}".format(
            args.train_file_dir, args.val_file_dir
        )
    )
    optimizer = optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )
    criterion = losses.__dict__["BCEDiceLoss"]().cuda()
    print("{} iterations per epoch".format(len(trainloader)))
    checkpoint_dir = "./checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_iou = 0
    best_dice = 0
    iter_num = 0
    max_epoch = 300
    max_iterations = len(trainloader) * max_epoch
    start_epoch = 0

    # Optionally resume training from the last checkpoint
    if args.continue_train:
        state_path = os.path.join(
            checkpoint_dir,
            "{}_state_{}.pth".format(args.model, args.train_file_dir.split(".")[0]),
        )
        if os.path.exists(state_path):
            # State checkpoint was created by this script, so it's safe to
            # load with weights_only=False even on PyTorch>=2.6
            checkpoint = torch.load(state_path, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            best_iou = checkpoint.get("best_iou", 0)
            best_dice = checkpoint.get("best_dice", 0)
            iter_num = checkpoint.get("iter_num", 0)
            print(
                "=> Resuming training from epoch {}/{} (best_iou={:.4f}, best_dice={:.4f}, iter_num={})".format(
                    start_epoch + 1, max_epoch, best_iou, best_dice, iter_num
                )
            )
        else:
            print(
                "=> No existing checkpoint found at {}. Starting from scratch.".format(
                    state_path
                )
            )

    for epoch_num in range(start_epoch, max_epoch):
        model.train()
        avg_meters = {
            "loss": AverageMeter(),
            "iou": AverageMeter(),
            "dice": AverageMeter(),
            "val_loss": AverageMeter(),
            "val_iou": AverageMeter(),
            "val_dice": AverageMeter(),
            "SE": AverageMeter(),
            "PC": AverageMeter(),
            "F1": AverageMeter(),
            "ACC": AverageMeter(),
        }
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)

            loss = criterion(outputs, label_batch)
            iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute learning rate schedule; clamp to avoid negative base
            # for the fractional power (which would yield a complex value).
            t = 1.0 - iter_num / max_iterations
            if t < 0.0:
                t = 0.0
            lr_ = float(base_lr * (t ** 0.9))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num = iter_num + 1
            avg_meters["loss"].update(loss.item(), volume_batch.size(0))
            avg_meters["iou"].update(iou, volume_batch.size(0))
            avg_meters["dice"].update(dice, volume_batch.size(0))

        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                input, target = sampled_batch["image"], sampled_batch["label"]
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = criterion(output, target)

                iou, dice, SE, PC, F1, _, ACC = iou_score(output, target)
                avg_meters["val_loss"].update(loss.item(), input.size(0))
                avg_meters["val_iou"].update(iou, input.size(0))
                avg_meters["val_dice"].update(dice, input.size(0))
                avg_meters["SE"].update(SE, input.size(0))
                avg_meters["PC"].update(PC, input.size(0))
                avg_meters["F1"].update(F1, input.size(0))
                avg_meters["ACC"].update(ACC, input.size(0))

        print(
            "epoch [%d/%d]  train_loss : %.4f, train_iou: %.4f "
            "- val_loss %.4f - val_iou %.4f - val_dice %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_ACC %.4f"
            % (
                epoch_num + 1,
                max_epoch,
                avg_meters["loss"].avg,
                avg_meters["iou"].avg,
                avg_meters["val_loss"].avg,
                avg_meters["val_iou"].avg,
                avg_meters["val_dice"].avg,
                avg_meters["SE"].avg,
                avg_meters["PC"].avg,
                avg_meters["F1"].avg,
                avg_meters["ACC"].avg,
            )
        )

        # Save best model based on validation IoU
        if avg_meters["val_iou"].avg > best_iou:
            torch.save(
                model.state_dict(),
                os.path.join(
                    checkpoint_dir,
                    "{}_model_{}.pth".format(
                        args.model, args.train_file_dir.split(".")[0]
                    ),
                ),
            )
            best_iou = avg_meters["val_iou"].avg
            best_dice = avg_meters["val_dice"].avg
            output_file_path = os.path.join(
                checkpoint_dir,
                "{}_model_{}.json".format(
                    args.model, args.train_file_dir.split(".")[0]
                ),
            )
            with open(output_file_path, "w") as file:
                json.dump(
                    {
                        **{k: v.avg for k, v in avg_meters.items()},
                        "epoch": epoch_num + 1,
                    },
                    file,
                    indent=4,
                )
            print("=> saved best model")

        # Save full training state every epoch for resuming with -c/--continue_train
        state_path = os.path.join(
            checkpoint_dir,
            "{}_state_{}.pth".format(args.model, args.train_file_dir.split(".")[0]),
        )
        torch.save(
            {
                "epoch": epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
                "best_dice": best_dice,
                "iter_num": iter_num,
            },
            state_path,
        )

    # Training finished successfully; remove the resume state checkpoint so
    # a future run starts fresh unless a new state is created.
    if os.path.exists(state_path):
        os.remove(state_path)

    return "Training Finished!"


if __name__ == "__main__":
    print(train(args))
