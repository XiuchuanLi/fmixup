import numpy as np
import argparse
import csv
import random

import torch
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
from utils.logger import get_logger


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


def approximate_gradient(x, label_ori, num_evals, beta, Distance):
    noise_shape = [num_evals] + list(x.shape)
    rv = torch.randn(*noise_shape)
    rv = rv / torch.sqrt(torch.sum(rv ** 2, dim=[1, 2, 3], keepdim=True))
    x_adv = clip(x + beta * rv)
    rv = (x_adv - x) / beta

    pred = norm_model(x_adv)
    Distance.extend([Distance[-1] for _ in range(num_evals)])
    success = (pred != label_ori)
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


def high_pass_filter(radius):
    flt = np.zeros([224, 224])
    for i in range(224):
        for j in range(224):
            if (i - 224 / 2) ** 2 + (j - 224 / 2) ** 2 + 1e-3 >= (224 * radius / 2) ** 2:
                flt[i, j] = 1
    return flt


def low_pass_filter(radius):
    flt = np.ones([224, 224])
    for i in range(224):
        for j in range(224):
            if (i - 224 / 2) ** 2 + (j - 224 / 2) ** 2 + 1e-3 >= (224 * radius / 2) ** 2:
                flt[i, j] = 0
    return flt


def filter(img, flt):
    fimg = np.fft.fft2(img)
    fimg = np.fft.fftshift(fimg)
    fimg = fimg * flt
    fimg = np.fft.ifftshift(fimg)
    iimg = np.fft.ifft2(fimg)
    iimg = iimg.real.astype(np.float32)
    return torch.tensor(iimg)


def frequency_binary_search(x_ori, x_adv, label_ori, Distance, r_low=0.0, r_high=1.42):
    assert norm_model(x_adv) != label_ori
    for _ in range(10):
        r_mid = (r_high + r_low) / 2.0
        x_candidate = clip(filter(x_ori, low_pass_filter(r_mid)) + filter(x_adv, high_pass_filter(r_mid)))
        pred = norm_model(x_candidate)
        if pred != label_ori:
            Distance.append(min(min(Distance), torch.norm(x_candidate - x_ori)))
            r_low = r_mid
        else:
            Distance.append(Distance[-1])
            r_high = r_mid
    x_candidate = clip(filter(x_ori, low_pass_filter(r_low)) + filter(x_adv, high_pass_filter(r_low)))
    if norm_model(x_candidate) == label_ori:
        return x_adv
    return x_candidate


def project(x_ori, x_adv, alpha):
    return (1-alpha) * x_ori + alpha * x_adv


def binary_search(x_ori, x_adv, label_ori, Distance):
    assert norm_model(x_adv) != label_ori
    dist_post = torch.norm(x_adv - x_ori)
    threshold = 1 / (torch.prod(torch.tensor(x_ori.shape)).item() ** 1.5)

    query = 0
    low, high = 0, 1
    while (high - low) > threshold:
        mid = (high + low) / 2.0
        mid_image = project(x_ori, x_adv, mid)
        pred = norm_model(mid_image)
        query += 1
        if pred != label_ori:
            Distance.append(min(min(Distance), torch.norm(mid_image - x_ori)))
            high = mid
        else:
            Distance.append(Distance[-1])
            low = mid
    x_adv = project(x_ori, x_adv, high)
    assert norm_model(x_adv) != label_ori
    return x_adv, dist_post, query


def geometric_progression_for_stepsize(x_ori, x_adv, label_ori, perturb, dist, cur_iter, Distance):
    alpha = dist / (cur_iter ** 0.5)
    query = 0
    while True:
        x_candidate = clip(x_adv + alpha * perturb)
        pred = norm_model(x_candidate)
        query += 1
        if pred != label_ori:
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
parser.add_argument("--valdir", "-v", type=str, default='../data/imagenet-val')
parser.add_argument("--max_query", "-q", type=int, default=5000)
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
num_clean = 200
clean_index = index[np.floor(np.linspace(0, index.shape[0] - 1, num_clean)).astype(np.int32)]
ds = datasets.ImageFolder(args.valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ]))
clean_examples = [ds[i] for i in clean_index]
# find start point for each clean example
start_index = []
random.seed(66)
for i in range(len(clean_examples)):
    while True:
        temp_index = random.randint(0, len(clean_examples)-1)
        if clean_examples[temp_index][1] != clean_examples[i][1]:
            start_index.append(temp_index)
            break

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape((3, 1, 1))
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape((3, 1, 1))

LOGGER.info(f"test")
for sample_id, (x_ori, label_ori) in enumerate(clean_examples):
    # start point
    query = 0
    x_adv = clean_examples[start_index[sample_id]][0]
    Distance = [torch.norm(x_adv - x_ori)]

    # initial binary search
    x_adv = frequency_binary_search(x_ori, x_adv, label_ori, Distance)
    query += 10
    x_adv, dist_post, count = binary_search(x_ori, x_adv, label_ori, Distance)
    query += count
    dist = torch.norm(x_adv - x_ori)
    cur_iter = 0
    # iteration
    while query <= args.max_query:
        cur_iter += 1

        # gradient estimation
        beta = select_beta(x_adv, dist_post, cur_iter)
        num_evals = int(100 * (cur_iter ** 0.5))
        perturb = approximate_gradient(x_adv, label_ori, num_evals, beta, Distance)
        query += num_evals

        # update adversarial example
        alpha, count = geometric_progression_for_stepsize(x_ori, x_adv, label_ori, perturb, dist, cur_iter, Distance)
        query += count
        x_adv = clip(x_adv + alpha * perturb)
        x_sadv, dist_post, count = binary_search(x_ori, x_adv, label_ori, Distance)
        query += count
        if cur_iter % 10 == 0:
            x_fadv = frequency_binary_search(x_ori, x_sadv, label_ori, Distance)
            query += 10
            if torch.norm(x_sadv - x_ori) > 0.2 + torch.norm(x_fadv - x_ori):
                x_sadv, _, count = binary_search(x_ori, x_fadv, label_ori, Distance)
                query += count
        x_adv = x_sadv

    LOGGER.info(f"{sample_id}, {query} queries")
    with open(f'revised_hsja_{args.model}.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([Distance[i].item() for i in range(0, args.max_query + 1, 100)])

