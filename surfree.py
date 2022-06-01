import numpy as np
import argparse
import csv
import random

import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
import torchvision.models as models
from utils.logger import get_logger
from utils.basis import get_mask, dct_8_8, idct_8_8, schmidt


def clip(x):
    x[x > 1] = 1
    x[x < 0] = 0
    return x


def norm_model(x):
    x = torch.unsqueeze(x, dim=0)
    x_norm = (x - mean) / std
    x_norm = x_norm.cuda()
    with torch.no_grad():
        logits = model(x_norm).cpu().squeeze()
        pred = torch.argmax(logits, dim=0)
    return pred.item()


def z_star_theta(x_ori, x_adv, theta, v):
    distance = torch.norm(x_adv - x_ori)
    u = (x_adv - x_ori) / distance
    theta = theta * np.pi / 180
    direction = np.cos(theta) * u + np.sin(theta) * v
    z = clip(x_ori + direction * distance * np.cos(theta))
    return z


def sign_search(x_ori, label_ori, x_adv, theta_max, v, Distance):
    taus = [1, -1]
    for i, tau in enumerate(taus):
        theta = tau * theta_max
        z = z_star_theta(x_ori, x_adv, theta, v)
        pred = norm_model(z)
        if pred != label_ori:
            Distance.append(min(min(Distance), torch.norm(z - x_ori)))
            return theta, i+1
        else:
            Distance.append(Distance[-1])
    return 0, 2


def binary_search(x_ori, label_ori, x_adv, theta_max, v, lower_theta, Distance, BS_max_iteration=7, BS_gamma=0.05):

    def get_alpha(theta):
        return 1 - np.cos(torch.tensor(theta * np.pi / 180))

    count = 0
    assert lower_theta != theta_max
    while True:
        upper_theta = lower_theta + np.sign(lower_theta) * theta_max
        z_upper_theta = z_star_theta(x_ori, x_adv, upper_theta, v)
        count += 1
        pred = norm_model(z_upper_theta)
        if pred != label_ori:
            lower_theta = upper_theta
            Distance.append(min(min(Distance), torch.norm(z_upper_theta - x_ori)))
        else:
            Distance.append(Distance[-1])
            break

    step = 0
    while step < BS_max_iteration and abs(get_alpha(upper_theta) - get_alpha(lower_theta)) > BS_gamma:
        mid_theta = (upper_theta + lower_theta) / 2
        z_mid_theta = z_star_theta(x_ori, x_adv, mid_theta, v)
        step += 1
        pred = norm_model(z_mid_theta)
        if pred != label_ori:
            lower_theta = mid_theta
            Distance.append(min(min(Distance), torch.norm(z_mid_theta - x_ori)))
        else:
            upper_theta = mid_theta
            Distance.append(Distance[-1])
    return lower_theta, count + step


def initialization(x_ori, label_ori, x_adv, Distance, steps=50):
    for i in range(steps):
        candidate = x_ori + (x_adv - x_ori) * i / steps
        pred = norm_model(candidate)
        if pred != label_ori:
            Distance.append(min(min(Distance), torch.norm(candidate - x_ori)))
            return candidate, i+1
        else:
            Distance.append(Distance[-1])
    return x_adv, 50


LOGGER = get_logger(__name__, level="DEBUG")
parser = argparse.ArgumentParser()
parser.add_argument("--valdir", "-v", type=str, default='data/imagenet-val')
parser.add_argument("--max_query", "-q", type=int, default=500)
parser.add_argument("--model", "-m", type=str, default='resnet')
args = parser.parse_args()

LOGGER.info(f"Loading model")
if args.model == "resnet":
    model = models.resnet50(pretrained=True).cuda()
    index = np.genfromtxt('utils/clean_resnet.csv', delimiter=',').astype(np.int32)
elif args.model == "mobilenet":
    model = models.mobilenet_v2(pretrained=True).cuda()
    index = np.genfromtxt('utils/clean_mobilenet.csv', delimiter=',').astype(np.int32)
else:
    raise ValueError("Invalid model")
model.eval()

LOGGER.info(f"Loading data")
num_clean, num_reference = 500, args.max_query
random.seed(66)
clean_index = index[np.floor(np.linspace(0, index.shape[0] - 1, num_clean)).astype(np.int32)]  # select clean examples
reference_index = np.setdiff1d(np.arange(50000), clean_index)  # remove clean examples
reference_index = np.random.choice(reference_index, num_reference, replace=False).astype(np.int32)  # select reference examples
ds = datasets.ImageFolder(args.valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ]))

ds_clean = [ds[i] for i in clean_index]
ds_reference = [ds[i] for i in reference_index]
del ds

# for fair comparison with f-attack, we use the closest reference example as the start point
start_index = []
for i in range(len(ds_clean)):
    distance = []
    for j in range(len(ds_reference)):
        distance.append(torch.sum((ds_clean[i][0] - ds_reference[j][0]) ** 2).item())
    rank = [index for index, value in sorted(list(enumerate(distance)), key=lambda x: x[1])]
    for r in rank:
        if ds_reference[r][1] != ds_clean[i][1]:
            start_index.append(r)
            break

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape((3, 1, 1))
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape((3, 1, 1))

for sample_id, (x_ori, label_ori) in enumerate(ds_clean):
    if sample_id < 201:
        continue
    theta_max = 30
    n_ortho = 100
    rho = 0.95
    query = 0
    x_adv = ds_reference[start_index[sample_id]][0]
    Distance = [torch.norm(x_adv - x_ori)]
    x_adv, count = initialization(x_ori, label_ori, x_adv, Distance)
    query += count

    mask = get_mask(0.5)
    fx_ori = dct_8_8(x_ori, mask)
    fx_ori = torch.tanh(fx_ori)

    LOGGER.info(f'{sample_id}')
    perturb = x_adv - x_ori
    perturb /= torch.norm(perturb)
    directions_ortho = torch.unsqueeze(perturb, dim=0)
    while query <= args.max_query:
        theta = 0
        while theta == 0:
            prob = (torch.randint(0, 3, fx_ori.shape) - 1).float()
            v = idct_8_8(fx_ori * prob) + torch.randn(x_ori.shape) * 0.01
            v = schmidt(v, directions_ortho)
            directions_ortho = torch.cat([directions_ortho, torch.unsqueeze(v, dim=0)], dim=0)
            if len(directions_ortho) > n_ortho + 1:
                directions_ortho = torch.cat([directions_ortho[:1], directions_ortho[n_ortho:]], dim=0)
            new_theta, count = sign_search(x_ori, label_ori, x_adv, theta_max, v, Distance)
            query += count
            if new_theta == 0:
                theta_max = theta_max * rho
            else:
                theta_max = theta_max / rho
                theta = new_theta

        theta, count = binary_search(x_ori, label_ori, x_adv, theta_max, v, theta, Distance)
        query += count
        candidate = z_star_theta(x_ori, x_adv, theta, v)
        if torch.norm(candidate - x_ori) < torch.norm(x_adv - x_ori):
            x_adv = candidate

    Distance = [elem.item() for elem in Distance]
    with open(f'surfree_{args.model}.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([Distance[0], Distance[100], Distance[200], Distance[300], Distance[400], Distance[500]])

