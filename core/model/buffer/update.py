import numpy as np
import torch
import copy
from torch.utils.data import DataLoader

EPSILON = 1e-8

def random_update(datasets, buffer):

    images = np.array(datasets.images + buffer.images)
    labels = np.array(datasets.labels + buffer.labels)
    # images = np.array(images)
    # labels = np.array(labels)
    perm = np.random.permutation(len(labels))

    images, labels = images[perm[:buffer.buffer_size]], labels[perm[:buffer.buffer_size]]

    buffer.images = images.tolist()
    buffer.labels = labels.tolist()

def hearding_update(datasets, buffer, feature_extractor, device):
    per_classes = buffer.buffer_size // buffer.total_classes

    selected_images, selected_labels = [], []
    images = np.array(datasets.images + buffer.images)
    labels = np.array(datasets.labels + buffer.labels)

    for cls in range(buffer.total_classes):
        print("Construct examplars for class {}".format(cls))
        cls_images_idx = np.where(labels == cls)
        cls_images, cls_labels = images[cls_images_idx], labels[cls_images_idx]

        cls_selected_images, cls_selected_labels = construct_examplar(copy.copy(datasets), cls_images, cls_labels, feature_extractor, per_classes, device)
        selected_images.extend(cls_selected_images)
        selected_labels.extend(cls_selected_labels)


    buffer.images, buffer.labels = selected_images, selected_labels

def herding_update_unified(datasets, buffer, feature_extractor, device, per_classes, start_cls_idx, end_cls_idx,test_trsfs):
    selected_images, selected_labels = [], []
    images = np.array(datasets.images)
    labels = np.array(datasets.labels)

    for cls in range(start_cls_idx, end_cls_idx):
        print("Construct examplars for class {}".format(cls))
        cls_images_idx = np.where(labels == cls)
        cls_images, cls_labels = images[cls_images_idx], labels[cls_images_idx]

        cls_selected_images, cls_selected_labels = construct_examplar_foster(copy.copy(datasets), cls_images, cls_labels,
                                                                      feature_extractor, per_classes, device, test_trsfs)
        selected_images.extend(cls_selected_images)
        selected_labels.extend(cls_selected_labels)

    buffer.images, buffer.labels = buffer.images+selected_images, buffer.labels+selected_labels
    print("buffer length{}".format(len(buffer.images)))

def construct_examplar_foster(datasets, images, labels, feature_extractor, per_classes, device, test_trsfs):

    if len(images) <= per_classes:
        return images, labels

    datasets.images, datasets.labels = images, labels
    datasets.trfms = test_trsfs
    dataloader = DataLoader(datasets, shuffle=False, batch_size=64, drop_last=False, num_workers=4)

    with torch.no_grad():
        features = []
        for data in dataloader:
            imgs = data['image'].to(device)
            feature = [convnet(imgs)["features"] for convnet in feature_extractor]
            feature = tensor2numpy(torch.cat(feature,1))
            # features.append(feature)
            features.append(feature)

    features = np.concatenate(features)
    features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

    selected_images, selected_labels = [], []
    selected_features = []
    class_mean = np.mean(features, axis=0)

    for k in range(1, per_classes + 1):
        S = np.sum(selected_features, axis=0)

        mu_p = (S + features) / k
        i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

        selected_images.append(images[i])
        selected_labels.append(labels[i])
        selected_features.append(features[i])

        features = np.delete(features, i, axis=0)
        images = np.delete(images, i)
        labels = np.delete(labels, i)

    return selected_images, selected_labels


def construct_examplar(datasets, images, labels, feature_extractor, per_classes, device):
    if len(images) <= per_classes:
        return images, labels
    
    datasets.images, datasets.labels = images, labels
    dataloader = DataLoader(datasets, shuffle = False, batch_size = 32, drop_last = False)

    with torch.no_grad():
        features = []
        for data in dataloader:
            imgs = data['image'].to(device)
            features.append(feature_extractor(imgs)['features'].cpu().numpy().tolist())

    features = np.concatenate(features)
    selected_images, selected_labels = [], []
    selected_features = []
    class_mean = np.mean(features, axis=0)

    for k in range(1, per_classes+1):
        if len(selected_features) == 0:
            S = np.zeros_like(features[0])
        else:
            S = np.mean(np.array(selected_features), axis=0)


        mu_p = (S + features) / k
        i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

        selected_images.append(images[i])
        selected_labels.append(labels[i])
        selected_features.append(features[i])

        features = np.delete(features, i, axis=0) 
        images = np.delete(images, i)
        labels = np.delete(labels, i)

    return selected_images, selected_labels


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()
    
        
