import os 

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import src.resnet50 as resnet_models

if __name__ == "__main__":
    # initialize data
    data_path = "/mnt/qb/work/berens/smueller93/"
    pretrained_weights = "/mnt/qb/work/berens/smueller93/swav/experiments/cifar10_500ep_bs512_pretrain/checkpoints/ckp-499.pth"
    batch_size = 512
    num_images = 4096
    experiment_path = "/mnt/qb/work/berens/smueller93/swav/experiments/cifar10_500ep_bs512_pretrain/eval_linear"

    train_dataset = datasets.CIFAR10(data_path, train=True)
    val_dataset = datasets.CIFAR10(data_path, train=False)
    tr_normalize = transforms.Normalize(
        mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
    )
    train_dataset.transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        tr_normalize,
    ])
    val_dataset.transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        tr_normalize,
    ])

    train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset, list(range(num_images))),
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_dataset, list(range(num_images))),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )

    # load model
    # build model
    model = resnet_models.__dict__["resnet50"](output_dim=128, eval_mode=False, nmb_prototypes=30)
    model = model.cuda()
    model = model.eval()

    # load weights
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cuda:0")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                print('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                print('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        print("Load pretrained model with msg: {}".format(msg))
    else:
        print("No pretrained weights found => training with random weights")

    preds_2048_train = torch.empty((num_images, 2048), dtype=torch.float32)
    preds_128_train = torch.empty((num_images, 128), dtype=torch.float32)
    prototypes_train = torch.empty((num_images, 30), dtype=torch.float32)
    labels_train = torch.empty((num_images), dtype=torch.int32)
    for i, (images, labels) in enumerate(train_loader):
        start = i*batch_size 
        stop = (i+1)*batch_size
        print(i, start, stop)
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            preds = model(images)
            images = images.cuda(non_blocking=True)
            preds_128, protos = model.forward(images)
            preds_2048 = model.forward_backbone(images)
        preds_128_train[start : stop] = preds_128
        prototypes_train[start : stop] = protos
        labels_train[start : stop] = labels
        preds_2048_train[start : stop] = preds_2048
    
    torch.save(
        preds_2048_train, 
        os.path.join(experiment_path, "preds_2048_train.pt")
    )

    torch.save(
        preds_128_train, 
        os.path.join(experiment_path, "preds_128_train.pt")
    )

    torch.save(
        prototypes_train, 
        os.path.join(experiment_path, "prototypes_train.pt")
    )

    torch.save(
        labels_train, 
        os.path.join(experiment_path, "labels_train.pt")
)