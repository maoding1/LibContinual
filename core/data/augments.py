from torchvision import transforms
from .autoaugment import *
from .cutout import *
from .randaugment import *

CJ_DICT = {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4}


def get_augment_method(
    config,
    mode,
):
    """Return the corresponding augmentation method according to the setting.

    + Use `ColorJitter` and `RandomHorizontalFlip` when not setting `augment_method` or using `NormalAug`.
    + Use `ImageNetPolicy()`when using `AutoAugment`.
    + Use `Cutout()`when using `Cutout`.
    + Use `RandAugment()`when using `RandAugment`.
    + Use `CenterCrop` and `RandomHorizontalFlip` when using `AutoAugment`.
    + Users can add their own augment method in this function.

    Args:
        config (dict): A LFS setting dict
        mode (str): mode in train/test/val

    Returns:
        list: A list of specific transforms.
    """
    # if mode == "train" and config["augment"]:
    #     # Add user's trfms here
    #     if "augment_method" not in config or config["augment_method"] == "NormalAug":
    #         trfms_list = get_default_image_size_trfms(config["image_size"])
    #         trfms_list += [
    #             transforms.ColorJitter(**CJ_DICT),
    #             transforms.RandomHorizontalFlip(),
    #         ]
    #     elif config["augment_method"] == "AutoAugment":
    #         trfms_list = get_default_image_size_trfms(config["image_size"])
    #         trfms_list += [ImageNetPolicy()]
    #     elif config["augment_method"] == "Cutout":
    #         trfms_list = get_default_image_size_trfms(config["image_size"])
    #         trfms_list += [Cutout()]
    #     elif config["augment_method"] == "RandAugment":
    #         trfms_list = get_default_image_size_trfms(config["image_size"])
    #         trfms_list += [RandAugment()]
    #     elif config["augment_method"] == "MTLAugment":  
    #         trfms_list = get_default_image_size_trfms(config["image_size"])
    #         # https://github.com/yaoyao-liu/meta-transfer-learning/blob/fe189c96797446b54a0ae1c908f8d92a6d3cb831/pytorch/dataloader/dataset_loader.py#L60
    #         trfms_list += [transforms.CenterCrop(80), transforms.RandomHorizontalFlip()]
    #     elif config["augment_method"] == "DeepBdcAugment":
    #         # https://github.com/Fei-Long121/DeepBDC/blob/main/data/datamgr.py#23
    #         trfms_list = [
    #             transforms.RandomResizedCrop(config["image_size"]),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ColorJitter(**CJ_DICT),
    #         ]
    #     elif config["augment_method"] == "S2M2Augment":
    #         trfms_list = [
    #             transforms.RandomResizedCrop(config["image_size"]),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ColorJitter(**CJ_DICT),
    #         ]
    #     else:
    #         trfms_list = get_default_image_size_trfms(config["image_size"])
    #         trfms_list += [
    #             transforms.ColorJitter(**CJ_DICT),
    #             transforms.RandomHorizontalFlip(),
    #         ]
            
    # else:
    #     if config["image_size"] == 224:
    #         trfms_list = [
    #             transforms.Resize((256, 256)),
    #             transforms.CenterCrop((224, 224)),
    #         ]
    #     elif config["image_size"] == 84:
    #         trfms_list = [
    #             transforms.Resize((96, 96)),
    #             transforms.CenterCrop((84, 84)),
    #         ]
    #     # for MTL -> alternative solution: use avgpool(ks=11)
    #     elif config["image_size"] == 80:
    #         trfms_list = [
    #             transforms.Resize((92, 92)),
    #             transforms.CenterCrop((80, 80)),
    #         ]
    #     else:
    #         raise RuntimeError
    trfms_list = []
    if mode == "train":
        trfms_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
    ]
    elif mode == "test":
        trfms_list = [transforms.ToTensor()]
    return trfms_list

def get_default_image_size_trfms(image_size):
    """ Return the uniform transforms for image_size """
    if image_size == 224:
        trfms = [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
        ]
    elif image_size == 84:
        trfms = [
            transforms.Resize((96, 96)),
            transforms.RandomCrop((84, 84)),
        ]
    # for MTL -> alternative solution: use avgpool(ks=11)
    elif image_size == 80:
        # MTL use another MEAN and STD
        trfms = [
            transforms.Resize((92, 92)),
            transforms.RandomResizedCrop(88),
            transforms.CenterCrop((80, 80)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        raise RuntimeError
    return trfms

class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img