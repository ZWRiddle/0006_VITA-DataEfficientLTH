import argparse
import utils
import random
import numpy as np
import torch
from torchvision import transforms, datasets
import augment
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
import models
import copy
import os
import json
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import tifffile
import torchsat.transforms.transforms_cls as transforms_sat
import datetime
import utils_custom as custom
import pruning_init
custom.check.counter = 0

# helper functions...
def line():
    print(f"|----------|----------|----------|----------|")

def timeline():
    line()
    print(datetime.datetime.now())

def linechecktime():
    # line()
    custom.check()
    print(datetime.datetime.now())



# Define a dictionary containing all preset Networks in "models"
NETWORKS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "fresnet18": models.resnet18,
    "fresnet34": models.resnet34,
    "fresnet50": models.resnet50,
    "fresnet101": models.resnet101,
    "fresnet152": models.resnet152,
    "hresnet18": models.resnet18,
    "hresnet34": models.resnet34,
    "hresnet50": models.resnet50,
    "hresnet101": models.resnet101,
    "hresnet152": models.resnet152,
    "vgg11": models.vgg11,
    "vgg13": models.vgg13,
    "vgg16": models.vgg16,
    "vgg19": models.vgg19,
    "mobilenet": models.MobileNetV2,
    "vit": models.VisionTransformer,
}

# List the Implemented Strategies
practical_strategies = ["ERK", "ERK+", "Uni", "Uni+", "SNIP","SNIP+","GraSP","GraSP+"]

# # Define Long tailed distribution, currently not needed
# def long_tailed_dist(......

# # Define a FewShot Class for specific datasets, currently not needed
# class FewShot(......

# Define the main class, Trainer

class Trainer:

    def __init__(self, args):

        # settings about reading args
        self.args = args
        self.out_dir = args.out_dir

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        # settings about main_thread _for_ distributed learning
        self.device, local_rank = utils.setup_device(args.dist)
        if args.dist:
            self.main_thread = True if local_rank == 0 else False
        else:
            self.main_thread = True
        if self.main_thread:
            print(f"\nsetting up device, distributed = {args.dist}")
        print(f" | {self.device}")

        # define transform t *** base_aug
        ## This part is simplified for only cifar10, and not with "vit"
        if args.dset != "cifar10":
            raise NotImplementedError(f"args.dset = {args.dset} not implemented.\nThis code only supports cifar10")
            exit()
        t = [# transforms.RandomCrop(32, padding=4),  # transforms.RandomHorizontalFlip(),
            ]
            
        # set augmentation (contrast, rand, auto, custom, blur, cutout)
       
        if args.contrast_aug:
            t.extend(
                [
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                0.8 * args.color_jitter_strength,
                                0.8 * args.color_jitter_strength,
                                0.8 * args.color_jitter_strength,
                                0.2 * args.color_jitter_strength,
                            )
                        ],
                        p=args.color_jitter_prob,
                    ),
                    transforms.RandomGrayscale(p=args.gray_prob),
                ]
            )
        if args.rand_aug:
            t.extend(
                [
                    augment.RandomAugment(args.n_rand_aug),
                ]
            )
        if args.auto_aug:
            t = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            t.extend([
                    augment.Policy(policy=args.auto_aug_policy),
            ])
        if args.custom_aug:
            t.extend(
                [
                    augment.ToNumpy(),
                    augment.CustomAugment.augment_image,
                    transforms.ToPILImage(),
                ]
            )
        if args.blur:
            t.extend(
                [
                    transforms.RandomApply(
                        [augment.GaussianBlur(args.blur_sigma)], p=args.blur_prob
                    ),
                ]
            )
        if args.cutout:
            t.extend(
                [
                    augment.Cutout(cut_len=args.cut_len),
                ]
            )
        
        # set normalize

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        # define train_transform, val_transform

        train_transform = transforms.Compose(
                [
                    *t,
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        
        if args.pretrained and "vit" in args.net:
            val_transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            val_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        
        # define train_dset, val_dset

        train_dset = datasets.CIFAR10(
            root=args.data_root, train=True, transform=train_transform, download=True
        )
        val_dset = datasets.CIFAR10(
            root=args.data_root, train=False, transform=val_transform, download=True
        )

        # train_test_split on data_size and data_size_2 --> data_indx --> train_sampler --> train_loader
        
        if 0 < args.data_size <= 1:
            data_size = args.data_size
        else:
            raise ValueError(f"args.datasize must be a float in (0,1], it is now )",args.data_size)
        
        if 0 < args.data_size_2 <= 1:
            data_size_2 = args.data_size_2
        else:
            raise ValueError(f"args.datasize_2 must be a float in (0,1], it is now )",args.data_size_2)

        if data_size == 1:
            data_indx = np.arange(len(train_dset))
        else:
            _, data_indx = train_test_split(
                np.arange(len(train_dset)),
                test_size=data_size,
                shuffle=True,
                stratify=train_dset.targets,
            )

        if data_size_2 == 1:
                data_indx_2 = np.arange(len(train_dset))
        else:
            _, data_indx_2 = train_test_split(
                        np.arange(len(train_dset)),
                        test_size=data_size_2,
                        shuffle=True, stratify=train_dset.targets,)

        if self.main_thread:
            linechecktime()
            print(f"setting up dataset, train: {len(data_indx)}, val: {len(val_dset)}")
            print(f"setting up dataset_2, train: {len(data_indx_2)}, val: {len(val_dset)}")
        train_sampler = SubsetRandomSampler(data_indx)
        self.train_loader = DataLoader(
            train_dset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.n_workers,
        )
        train_sampler_2 = SubsetRandomSampler(data_indx_2)
        self.train_loader_2 = DataLoader(
            train_dset,
            batch_size=args.batch_size,
            sampler=train_sampler_2,
            num_workers=args.n_workers,
        )

        # final bit, define val_loader, criterion, cos_criterion, metric_meter      

        self.val_loader = DataLoader(
            val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        if args.cos_criterion:
            self.cos_criterion = torch.nn.CosineEmbeddingLoss()
        self.metric_meter = utils.AvgMeter()  

    def train_epoch(self):
        timeline()
        self.metric_meter.reset()
        self.model.train()
        for indx, (img, target) in enumerate(self.train_loader):
            img, target = img.to(self.device).float(), target.to(self.device)

            if self.args.adv_prop:
                pred, adv_pred = self.model(img, target, adv_prop=True)
                loss = (self.criterion(pred, target) + self.criterion(adv_pred, target)) / 2
            else:
                if self.args.cos_linear == True or self.args.tvmf_linear == True:
                    pred, loss = self.model(img, target)
                else:
                    pred = self.model(img)
                    loss = self.criterion(pred, target)
                    if self.args.cos_criterion:
                        one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[-1])
                        loss = 0.1 * loss + self.cos_criterion(
                            pred, one_hot, torch.ones(target.shape).to(target.device)
                        )

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            pred_cls = pred.argmax(dim=1)
            acc = pred_cls.eq(target.view_as(pred_cls)).sum().item() / img.shape[0]

            metrics = {"train loss": loss.item(), "train acc": acc}
            self.metric_meter.add(metrics)
            utils.pbar(indx / len(self.train_loader), msg=self.metric_meter.msg())
        utils.pbar(1, msg=self.metric_meter.msg())

    def train_epoch_2(self):
        timeline()
        self.metric_meter.reset()
        self.model.train()
        for indx, (img, target) in enumerate(self.train_loader_2):
            img, target = img.to(self.device).float(), target.to(self.device)

            if self.args.adv_prop:
                pred, adv_pred = self.model(img, target, adv_prop=True)
                loss = (self.criterion(pred, target) + self.criterion(adv_pred, target)) / 2
            else:
                if self.args.cos_linear == True or self.args.tvmf_linear == True:
                    pred, loss = self.model(img, target)
                else:
                    pred = self.model(img)
                    loss = self.criterion(pred, target)
                    if self.args.cos_criterion:
                        one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[-1])
                        loss = 0.1 * loss + self.cos_criterion(
                            pred, one_hot, torch.ones(target.shape).to(target.device)
                        )

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            pred_cls = pred.argmax(dim=1)
            acc = pred_cls.eq(target.view_as(pred_cls)).sum().item() / img.shape[0]

            metrics = {"train loss": loss.item(), "train acc": acc}
            self.metric_meter.add(metrics)
            utils.pbar(indx / len(self.train_loader_2), msg=self.metric_meter.msg())
        utils.pbar(1, msg=self.metric_meter.msg())

    @torch.no_grad()
    def eval(self):
        self.metric_meter.reset()
        self.model.eval()
        if self.args.adv_prop:
            self.model.apply(utils.to_clean)
        for indx, (img, target) in enumerate(self.val_loader):
            img, target = img.to(self.device).float(), target.to(self.device)

            if self.args.cos_linear == True or self.args.tvmf_linear == True:
                pred, loss = self.model(img, target)
            else:
                pred = self.model(img)
                loss = self.criterion(pred, target)

            pred_cls = pred.argmax(dim=1)
            acc = pred_cls.eq(target.view_as(pred_cls)).sum().item() / img.shape[0]

            metrics = {"val loss": loss.item(), "val acc": acc}
            self.metric_meter.add(metrics)
            utils.pbar(indx / len(self.val_loader), msg=self.metric_meter.msg())
        utils.pbar(1, msg=self.metric_meter.msg())
    

    def train_new_imp(self):
        # Define model...
        if self.args.dset == "cifar10":
            model = NETWORKS[self.args.net](
                n_cls=10, pre_conv="small", pretrained=self.args.pretrained
            )
        else:
            raise NotImplementedError(f"args.dset = {self.args.dset} not implemented.")
        if self.args.tvmf_linear:
            model.linear = resnet.TVMF(model.linear.weight.shape[1], model.linear.weight.shape[0])
        if self.args.adv_prop:
            utils.modify_bn(model)
            setattr(
                model,
                "attacker",
                utils.PGDAttacker(
                    self.args.attack_n_iter, self.args.attack_eps, self.args.attack_step_size, 0.2
                ),
            )
        self.model = model.to(self.device)
        # Define optim...
        self.optim = utils.setup_optim(self.args, self.model.parameters())
        self.optim, self.lr_sched = utils.setup_lr_sched(self.args, self.optim)

        # Checking out_dir and load_dir, setting init, prune_iter_start, and start_epoch
        if os.path.exists(os.path.join(self.args.out_dir, "last_imp.ckpt")):
            if self.args.resume == False:
                raise ValueError(
                    f"directory {self.args.out_dir} already exists, change output directory or use --resume argument"
                )
            ckpt = torch.load(
                os.path.join(self.args.out_dir, "last_imp.ckpt"), map_location=self.device
            )
            self.model.load_state_dict(ckpt["model"])
            self.optim.load_state_dict(ckpt["optim"])
            self.lr_sched.load_state_dict(ckpt["lr_sched"])
            init = ckpt["init"]
            prune_iter_start = ckpt["iter"]
            start_epoch = ckpt["epoch"]
            print(f"\nresuming imp training from iter = {prune_iter_start}, epoch = {start_epoch}")
        else:
            if self.args.resume == True:
                raise ValueError(
                    f"args.resume = true, but no checkpoint found in {self.args.out_dir}"
                )
            os.makedirs(self.args.out_dir, exist_ok=True)
            with open(os.path.join(self.args.out_dir, "args_imp.txt"), "w") as f:
                json.dump(self.args.__dict__, f, indent=4)
            prune_iter_start = 0
            start_epoch = 0
            init = copy.deepcopy(self.model.state_dict())
            print(f"\nstarting imp training from scratch")
    
        # Begin for-loop of iterations...
        for iter in range(prune_iter_start, self.args.pruning_iters):
            if self.main_thread:
                linechecktime()
                print(f"pruning state: {iter}")
                print("remaining weight at start of iteration = ", utils.check_sparsity(self.model))
            
            # Begin for-loop of epochs...
            best_train, best_val = 0, 0
            for epoch in range(start_epoch, self.args.epochs):
                if self.main_thread:
                    print(
                        f"\niteration: {iter}, epoch: {epoch}, best train: {round(best_train, 5)}, best val: {round(best_val, 5)}, lr: {round(self.optim.param_groups[0]['lr'], 5)}"
                    )

                self.train_epoch()

                # Resetting init at rewinding epoch (default 2)
                if (
                    iter == 0
                    and self.args.rewind_type == "epoch"
                    and epoch == self.args.rewind_epoch
                ):
                    init = copy.deepcopy(self.model.state_dict())

                if self.main_thread:
                    train_metrics = self.metric_meter.get()
                    self.eval()
                    val_metrics = self.metric_meter.get()

                    # Check & print train acc improvement
                    if train_metrics["train acc"] > best_train:
                        print(
                            "\x1b[34m"
                            + f"train acc improved from {round(best_train, 5)} to {round(train_metrics['train acc'], 5)}"
                            + "\033[0m"
                        )
                        best_train = train_metrics["train acc"]

                    # Check & print val acc improvement, update best_imp_{iter}.ckpt
                    if val_metrics["val acc"] > best_val:
                        print(
                            "\x1b[33m"
                            + f"val acc improved from {round(best_val, 5)} to {round(val_metrics['val acc'], 5)}"
                            + "\033[0m"
                        )
                        best_val = val_metrics["val acc"]
                        torch.save(
                            {"model": self.model.state_dict(), "init": init},
                            os.path.join(self.args.out_dir, f"best_imp_{iter}.ckpt"),
                        )

                    # Saving model to last_imp.ckpt & last_imp_{iter}.ckpt
                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "optim": self.optim.state_dict(),
                            "lr_sched": self.lr_sched.state_dict(),
                            "init": init,
                            "iter": iter,
                            "epoch": epoch,
                        },
                        os.path.join(self.args.out_dir, "last_imp.ckpt"),
                    )
                    torch.save(
                        {"model": self.model.state_dict(), "init": init},
                        os.path.join(self.args.out_dir, f"last_imp_{iter}.ckpt"),
                    )

                if epoch < self.args.warmup_epochs:
                    self.optim.param_groups[0]["lr"] = (
                        epoch / self.args.warmup_epochs * self.args.lr
                    )
                else:
                    self.lr_sched.step()
            
            # Useless code about pt rewinding...
            if iter == 0 and self.args.rewind_type == "pt":
                init = torch.load(
                    os.path.join(self.args.out_dir, f"best_imp_{iter}.ckpt"),
                    map_location=self.device,
                )["model"]

            # l1 prune...
            linechecktime()
            if self.args.prune_type == "l1":
                if "vit" in self.args.net:
                    utils.l1_prune_vit(self.model, self.args.prune_rate, self.args.prune_ff_only)
                else:
                    utils.l1_prune(self.model, self.args.prune_rate)
            else:
                raise NotImplementedError(
                    f"args.prune_type = {self.args.prune_type} is not implemented, should be l1 only."
                )

            # Save curr_mask...
            curr_mask = utils.extract_mask(self.model.state_dict())
            custom.extract_layer_sparsity(curr_mask)

            # Remove prune, reset weights to init, then re-apply curr_mask...
            if "vit" in self.args.net:
                utils.remove_prune_vit(self.model, self.args.prune_ff_only)
            else:
                utils.remove_prune(self.model)
            self.model.load_state_dict(init)
            if "vit" in self.args.net:
                utils.mask_prune_vit(self.model, curr_mask, self.args.prune_ff_only)
            else:
                utils.mask_prune(self.model, curr_mask)

            # Reset optim, lr_sched, and start_epoch
            self.optim = utils.setup_optim(self.args, self.model.parameters())
            self.optim, self.lr_sched = utils.setup_lr_sched(self.args, self.optim)
            start_epoch = 0
            if self.args.rewind_type:
                for _ in range(self.args.rewind_epoch):
                    self.lr_sched.step()
        
            if self.main_thread:
                if iter == (self.args.pruning_iters-1):
                    linechecktime()
                    print(f"pruning state {iter} finished.")
                    print("remaining weight at end of iteration = ", utils.check_sparsity(self.model))
            
            # Print best_val at end of iteration:
            if self.main_thread:
                linechecktime()
                print(f"best_val = ",best_val)
            
            

        torch.save({
                            "model": self.model.state_dict(),
                            "optim": self.optim.state_dict(),
                            "lr_sched": self.lr_sched.state_dict(),
                            "init": init,
                            "iter": iter,
                            "epoch": epoch,
                    },
                    os.path.join(self.args.out_dir, "load_imp.ckpt"),)
            

    def train(self):

        # Define model...
        if self.args.dset == "cifar10":
            model = NETWORKS[self.args.net](
                n_cls=10, pre_conv="small", pretrained=self.args.pretrained
            )
        else:
            raise NotImplementedError(f"args.dset = {self.args.dset} not implemented.")
        if self.args.tvmf_linear:
            model.linear = resnet.TVMF(model.linear.weight.shape[1], model.linear.weight.shape[0])
        if self.args.adv_prop:
            utils.modify_bn(model)
            setattr(
                model,
                "attacker",
                utils.PGDAttacker(
                    self.args.attack_n_iter, self.args.attack_eps, self.args.attack_step_size, 0.2
                ),
            )
        self.model = model.to(self.device)
        # Define optim...
        self.optim = utils.setup_optim(self.args, self.model.parameters())
        self.optim, self.lr_sched = utils.setup_lr_sched(self.args, self.optim)

        # Initialize init...
        init = copy.deepcopy(self.model.state_dict())

        # If need to load...
        if args.prune_strategy == "full_dset":   
            # Checking out_dir and load_dir, setting init, prune_iter_start, and start_epoch
            if os.path.exists(os.path.join(self.args.load_dir, "load_imp.ckpt")):          
                ckpt = torch.load(
                    os.path.join(self.args.load_dir, "load_imp.ckpt"), map_location=self.device
                )
                # self.model.load_state_dict(ckpt["model"])
                self.optim.load_state_dict(ckpt["optim"])
                self.lr_sched.load_state_dict(ckpt["lr_sched"])

                if not os.path.exists(self.args.out_dir):
                    os.makedirs(self.args.out_dir
                    )
                with open(os.path.join(self.args.out_dir, "args_imp.txt"), "w") as f:
                    json.dump(self.args.__dict__, f, indent=4
                    )
                prune_iter_start = 0
                start_epoch = 0   

                # Loading layerwise_sparsity...
                linechecktime()
                print(f"\nextracting layerwise_sparsity from loaded ckpt...\n")
                layerwise_sparsity = custom.extract_layer_sparsity(custom.load_mask(ckpt)) 

            else:
                raise NotImplementedError(f"No load_imp.ckpt found in {self.args.load_dir}."
                )
        elif args.prune_strategy in practical_strategies:
            os.makedirs(self.args.out_dir, exist_ok=True)
            with open(os.path.join(self.args.out_dir, "args_imp.txt"), "w") as f:
                json.dump(self.args.__dict__, f, indent=4)
            prune_iter_start = 0
            start_epoch = 0
            init = copy.deepcopy(self.model.state_dict())
            print(f"\nstarting training from scratch")

            # Setting layer wise sparsity according to prune_strategy:
            linechecktime()
            final_sparsity = 1-pow((1-self.args.prune_rate), self.args.pruning_iters)
            print(
                f"\nprune_rate = {self.args.prune_rate}, iters = {self.args.pruning_iters}, final_sparsity = {final_sparsity}.\n"
                )
            if args.prune_strategy == 'SNIP':
                print('Initialize by SNIP')
                layerwise_sparsity = pruning_init.SNIP(self.model, final_sparsity, self.train_loader_2, self.device)
                

            if args.prune_strategy == 'SNIP+':
                print('Initialize by SNIP+')
                layerwise_sparsity = pruning_init.SNIP(self.model, final_sparsity, self.train_loader_2, self.device)

                ### Dirty patch, Manually Conpensate Small Layers:
                tiny_layers = [0,7]
                small_layers = [1,2,3,4,5,6,8,9,10,12,17]
                for layer in tiny_layers:
                    if layerwise_sparsity[layer] < 0.2:
                        layerwise_sparsity[layer] = 0.2
                for layer in small_layers:
                    if layerwise_sparsity[layer] < 1-final_sparsity:
                        layerwise_sparsity[layer] = 1-final_sparsity
                
                
                print(f"Layerwise Density after patching: ", layerwise_sparsity)

            if args.prune_strategy == 'GraSP':
                print('Initialize by GraSP')
                layerwise_sparsity = pruning_init.GraSP(self.model, final_sparsity, self.train_loader_2, self.device)
                # print(layerwise_sparsity)
            
            if args.prune_strategy == 'GraSP+':
                print('Initialize by GraSP+')
                layerwise_sparsity = pruning_init.GraSP(self.model, final_sparsity, self.train_loader_2, self.device)

                # print(layerwise_sparsity)
                tiny_layers = [0,7]
                small_layers = [1,2,3,4,5,6,8,9,10,12,17]
                for layer in tiny_layers:
                    if layerwise_sparsity[layer] < 0.2:
                        layerwise_sparsity[layer] = 0.2
                for layer in small_layers:
                    if layerwise_sparsity[layer] < 1-final_sparsity:
                        layerwise_sparsity[layer] = 1-final_sparsity

            if args.prune_strategy == 'Uni+':
                print('Initialize by Uni+')
                layerwise_sparsity = []
                
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        layerwise_sparsity.append(1-final_sparsity)
                layerwise_sparsity[0] = 1
                print(layerwise_sparsity)
        
        # If load mask directly (Mask), or layerwise random prune (LWR), iters = 1
        if args.mode in ["Mask", "LWR"]:
            self.args.pruning_iters = 1

        # For layerwise random pruning:
            if args.mode == "LWR":
                # print("\t\tname\t\t|\t\t\tmodule[name]\t\t\t|\tlayerwise_sparsity[name]\t")
                i = 0
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        tojoin = [name,".weight_mask"]
                        # print(name, end = "\t|\t")
                        # print(module, end = "\t|\t")
                        if args.prune_strategy == 'full_dset':
                            # print(layerwise_sparsity["".join(tojoin)])
                            PR = 1 - layerwise_sparsity["".join(tojoin)]
                        elif args.prune_strategy in practical_strategies:
                            # print(layerwise_sparsity[i])
                            PR = 1 - layerwise_sparsity[i]
                            i += 1
                        utils.random_prune(module, PR)

                curr_mask = utils.extract_mask(self.model.state_dict())    
                custom.extract_layer_sparsity(curr_mask)       
                
                utils.remove_prune(self.model)
                self.model.load_state_dict(init)
                utils.mask_prune(self.model, curr_mask)

        # Begin for-loop of iterations...
        overall_best = 0
        for iter in range(prune_iter_start, self.args.pruning_iters):
            if self.main_thread and self.args.pruning_iters > 1:
                linechecktime()
                print(f"pruning state: {iter}")
                print("remaining weight at start of iteration = ", utils.check_sparsity(self.model))

            # If load mask directly...
            if args.mode == "Mask":
                # Define Mask...
                raise NotImplementedError(f"Mask mode not yet implemented!")
            
            
            # Begin for-loop of epochs...
            best_train, best_val = 0, 0
            for epoch in range(start_epoch, self.args.epochs):
                if self.main_thread:
                    print(
                        f"\niteration: {iter}, epoch: {epoch}, best train: {round(best_train, 5)}, best val: {round(best_val, 5)}, lr: {round(self.optim.param_groups[0]['lr'], 5)}"
                    )

                self.train_epoch_2()

                # Resetting init at rewinding epoch (default 2)
                if (
                    iter == 0
                    and self.args.rewind_type == "epoch"
                    and epoch == self.args.rewind_epoch
                ):
                    init = copy.deepcopy(self.model.state_dict())

                if self.main_thread:
                    train_metrics = self.metric_meter.get()
                    self.eval()
                    val_metrics = self.metric_meter.get()

                    # Check & print train acc improvement
                    if train_metrics["train acc"] > best_train:
                        print(
                            "\x1b[34m"
                            + f"train acc improved from {round(best_train, 5)} to {round(train_metrics['train acc'], 5)}"
                            + "\033[0m"
                        )
                        best_train = train_metrics["train acc"]

                    # Check & print val acc improvement, update best_imp_{iter}.ckpt
                    if val_metrics["val acc"] > best_val:
                        print(
                            "\x1b[33m"
                            + f"val acc improved from {round(best_val, 5)} to {round(val_metrics['val acc'], 5)}"
                            + "\033[0m"
                        )
                        best_val = val_metrics["val acc"]
                        torch.save(
                            {"model": self.model.state_dict(), "init": init},
                            os.path.join(self.args.out_dir, f"best_imp_{iter}.ckpt"),
                        )

                    # Saving model to last_imp.ckpt & last_imp_{iter}.ckpt
                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "optim": self.optim.state_dict(),
                            "lr_sched": self.lr_sched.state_dict(),
                            "init": init,
                            "iter": iter,
                            "epoch": epoch,
                        },
                        os.path.join(self.args.out_dir, "last_imp.ckpt"),
                    )
                    torch.save(
                        {"model": self.model.state_dict(), "init": init},
                        os.path.join(self.args.out_dir, f"last_imp_{iter}.ckpt"),
                    )

                if epoch < self.args.warmup_epochs:
                    self.optim.param_groups[0]["lr"] = (
                        epoch / self.args.warmup_epochs * self.args.lr
                    )
                else:
                    self.lr_sched.step()
            
            # Useless code about pt rewinding...
            if iter == 0 and self.args.rewind_type == "pt":
                init = torch.load(
                    os.path.join(self.args.out_dir, f"best_imp_{iter}.ckpt"),
                    map_location=self.device,
                )["model"]

            # For layerwise IMP:
            if args.mode == "imp" or args.mode == "IMP":
                i = 0
                linechecktime()
                # print("\t\tname\t\t|\t\t\tmodule[name]\t\t\t|\tlayerwise_sparsity[name]\t")
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        tojoin = [name,".weight_mask"]
                        # print(name, end = "\t|\t")
                        # print(module, end = "\t|\t")
                        # print(layerwise_sparsity["".join(tojoin)])
                        if args.prune_strategy == 'full_dset':
                            # print(layerwise_sparsity["".join(tojoin)])
                            PR = 1 - pow((layerwise_sparsity["".join(tojoin)]),1/self.args.pruning_iters)
                        elif args.prune_strategy in practical_strategies:
                            # print(layerwise_sparsity[i])
                            PR = 1-pow((layerwise_sparsity[i]),1/self.args.pruning_iters)
                            i += 1
                        utils.l1_prune(module, PR)
                print(f"\nextracting layerwise_sparsity...\n")
                # custom.extract_layer_sparsity(self.model.state_dict()) 

                curr_mask = utils.extract_mask(self.model.state_dict())           
                custom.extract_layer_sparsity(curr_mask)

                utils.remove_prune(self.model)
                self.model.load_state_dict(init)
                utils.mask_prune(self.model, curr_mask)
                self.optim = utils.setup_optim(self.args, self.model.parameters())
                self.optim, self.lr_sched = utils.setup_lr_sched(self.args, self.optim)
                start_epoch = 0
                if self.args.rewind_type:
                    for _ in range(self.args.rewind_epoch):
                        self.lr_sched.step()
            
            # Track sparsity at end of iteration:
            if self.main_thread:
                if self.args.pruning_iters > 1:
                    if iter == (self.args.pruning_iters-1):
                        linechecktime()
                        print(f"pruning state {iter} finished.")
                        print("remaining weight at end of iteration = ", utils.check_sparsity(self.model))
                elif self.args.pruning_iters == 1:
                    if iter == (self.args.pruning_iters-1):
                        linechecktime()
                        print(f"pruning finished.")
                        print("remaining weight at end of run = ", utils.check_sparsity(self.model))

            # Print best_val at end of iteration:
            if self.main_thread:
                linechecktime()
                print(f"best_val = ",best_val)

            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = utils.add_args(parser)
    args = parser.parse_args()
    utils.print_args(args)

    trainer = Trainer(args)

    if args.prune_strategy == "New":
        trainer.train_new_imp()
    else:
        trainer.train()


    # if args.dist:
    #     torch.distributed.destroy_process_group()