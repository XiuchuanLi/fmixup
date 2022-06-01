import numpy as np
import argparse
import csv
import random

import torch
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
from utils.logger import get_logger
from utils.bilinear import bilinear


def clip(x):
    x[x > 1] = 1
    x[x < 0] = 0
    return x


def norm_model(x):
    if x.dim() == 3:
        x = torch.unsqueeze(x, dim=0)
    x_norm = (x - mean) / std
    x_norm = x_norm.cuda()
    with torch.no_grad():
        logits = model(x_norm).cpu()
        pred = torch.argmax(logits, dim=1)
    return pred.squeeze()


def approximate_gradient(x, label_tar, num_evals, beta, Distance, current_query):
    bilinear_noise_index = np.arange(current_query, current_query+num_evals)
    rv = bilinear_noise[bilinear_noise_index]
    rv = rv / torch.sqrt(torch.sum(rv ** 2, dim=[1, 2, 3], keepdim=True))
    x_adv = clip(x + beta * rv)
    rv = (x_adv - x) / beta

    pred = norm_model(x_adv)
    Distance.extend([Distance[-1] for _ in range(num_evals)])
    success = (pred == label_tar)
    fval = 2 * success.float().reshape([num_evals, 1, 1, 1]) - 1.0

    if torch.mean(fval) == 1.0:
        gradf = torch.mean(rv, dim=0)
    elif torch.mean(fval) == -1.0:
        gradf = - torch.mean(rv, dim=0)
    else:
        fval -= torch.mean(fval)
        gradf = torch.mean(fval * rv, dim=0)

    gradf = gradf / torch.norm(gradf)
    return gradf


def project(x_ori, x_adv, alpha):
    return (1-alpha) * x_ori + alpha * x_adv


def binary_search(x_ori, x_adv, label_tar, Distance):
    assert norm_model(x_adv) == label_tar
    dist_post = torch.norm(x_adv - x_ori)
    threshold = 1 / (torch.prod(torch.tensor(x_ori.shape)).item() ** 1.5)

    query = 0
    low, high = 0, 1
    while (high - low) > threshold:
        mid = (high + low) / 2.0
        mid_image = project(x_ori, x_adv, mid)
        pred = norm_model(mid_image)
        query += 1
        if pred == label_tar:
            Distance.append(min(min(Distance), torch.norm(mid_image - x_ori)))
            high = mid
        else:
            Distance.append(Distance[-1])
            low = mid
    x_adv = project(x_ori, x_adv, high)
    assert norm_model(x_adv) == label_tar
    return x_adv, dist_post, query


def geometric_progression_for_stepsize(x_ori, x_adv, label_tar, perturb, dist, cur_iter, Distance):
    alpha = dist / (cur_iter ** 0.5)
    query = 0
    while True:
        x_candidate = clip(x_adv + alpha * perturb)
        pred = norm_model(x_candidate)
        query += 1
        if pred == label_tar:
            Distance.append(min(min(Distance), torch.norm(x_candidate - x_ori)))
            break
        else:
            Distance.append(Distance[-1])
        alpha /= 2.0
    return alpha, query


def select_beta(x_adv, dist_post, cur_iter):
    if cur_iter == 1:
        beta = 0.1
    else:
        beta = dist_post / torch.prod(torch.tensor(x_adv.shape)).item()
    return beta


LOGGER = get_logger(__name__, level="DEBUG")
parser = argparse.ArgumentParser()
parser.add_argument("--valdir", "-v", type=str, default='data/imagenet-val')
parser.add_argument("--max_query", "-q", type=int, default=5000)
parser.add_argument("--model", "-m", type=str, default='mobilenet')
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
num_clean = 200
clean_index = index[np.floor(np.linspace(0, index.shape[0] - 1, num_clean)).astype(np.int32)]
ds = datasets.ImageFolder(args.valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ]))
clean_examples = [ds[i] for i in clean_index]
# find start point for each clean example
start_index, target_label = [], []
random.seed(66)
for i in range(len(clean_examples)):
    while True:
        temp_index = random.randint(0, len(clean_examples)-1)
        if clean_examples[temp_index][1] != clean_examples[i][1]:
            start_index.append(temp_index)
            target_label.append(clean_examples[temp_index][1])
            break

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape((3, 1, 1))
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape((3, 1, 1))
LOGGER.info(f"creat bilinear noise")
bilinear_noise = bilinear(int(args.max_query * 1.2))

LOGGER.info(f"test")
for sample_id, (x_ori, _) in enumerate(clean_examples):
    # start point
    query = 0
    x_adv = clean_examples[start_index[sample_id]][0]
    label_tar = target_label[sample_id]
    Distance = [torch.norm(x_adv - x_ori)]

    # initial binary search
    x_adv, dist_post, count = binary_search(x_ori, x_adv, label_tar, Distance)
    query += count
    dist = torch.norm(x_adv - x_ori)
    cur_iter = 0
    # iteration
    while query <= args.max_query:
        cur_iter += 1

        # gradient estimation
        beta = select_beta(x_adv, dist_post, cur_iter)
        num_evals = int(100 * (cur_iter ** 0.5))
        perturb = approximate_gradient(x_adv, label_tar, num_evals, beta, Distance, query)
        query += num_evals

        # update adversarial example
        alpha, count = geometric_progression_for_stepsize(x_ori, x_adv, label_tar, perturb, dist, cur_iter, Distance)
        query += count
        x_adv = clip(x_adv + alpha * perturb)
        x_adv, dist_post, count = binary_search(x_ori, x_adv, label_tar, Distance)
        query += count

    LOGGER.info(f"{sample_id}, {query} queries")
    with open(f'vanilla_qeba_target_{args.model}.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([Distance[i].item() for i in range(0, args.max_query + 1, 100)])

