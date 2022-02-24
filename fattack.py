import argparse
import csv
import os.path
import numpy as np
from collections import deque
import hashlib

import torch
from torchvision import datasets
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from utils.logger import get_logger

LOGGER = get_logger(__name__, level="DEBUG")


def band_pass_filter(low, up):
    flt = np.zeros([3, 224, 224])
    for i in range(224):
        for j in range(224):
            if (224 * up / 2) ** 2 > (i - 224 / 2) ** 2 + (j - 224 / 2) ** 2 > (224 * low / 2) ** 2:
                flt[:, i, j] = 1
    return flt


def calculate_perturb(pass_clean, pass_reference, norm_perturb):
    norm_pass_clean = torch.norm(pass_clean)
    norm_pass_reference = torch.norm(pass_reference)
    cos_theta = torch.sum(pass_clean * pass_reference) / (norm_pass_clean * norm_pass_reference)

    a = 1
    b = -2 * norm_pass_clean * cos_theta
    c = norm_pass_clean ** 2 - norm_perturb ** 2
    delta = b ** 2 - 4 * a * c + 1e-6

    scale = ((-b + torch.sqrt(delta)) / 2) / norm_pass_reference
    return pass_reference * scale - pass_clean


def norm_model(x):
    x = (x - mean) / std
    x = x.cuda()
    with torch.no_grad():
        logits = model(x).cpu()
        prob = F.softmax(logits, dim=1)
        top2 = prob.topk(2)[0]
        ambiguities = top2[:, 0] - top2[:, 1]
    return torch.argmax(logits, dim=1), ambiguities


def write_log(log_file, row):
    if not os.path.exists(log_file):
        with open(log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(['index', 'queries', 'distortion'])
    with open(log_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(row)


def boundary_detection(ambiguities, n, k, epsilon_p):
    que = deque([])
    for i in range(len(ambiguities)):
        if ambiguities[i] < epsilon_p:
            que.append(1)
        else:
            que.append(0)
        if len(que) > n:
            que.popleft()
        if sum(que) >= k:
            return i + 1
    return -1


def fingerprint(img):
    img = img.flatten().numpy()
    img = ((img * 255) // 50).astype(np.int8)
    string = ''
    fingerprint = []
    for elem in img:
        string += str(elem)
    for i in range(0, len(string)-49):
        string_temp = string[i:i+50]
        m = hashlib.md5()
        m.update(string_temp.encode('utf-8'))
        fingerprint.append(int(m.hexdigest(), base=16))
    fingerprint = list(set(fingerprint))
    return set(sorted(fingerprint, reverse=True)[:50])


def blacklight(adv_img):
    buffer = [fingerprint(adv_img[0])]
    for current in range(1, len(adv_img)):
        buffer.append(fingerprint(adv_img[current]))
        for history in range(0, current):
            overlap = len(buffer[current] & buffer[history])
            if overlap > 25:
                return current + 1
    return -1


parser = argparse.ArgumentParser()
parser.add_argument("--valdir", "-v", type=str, default='../data/imagenet-val')
parser.add_argument("--radius", "-r", type=float, default=0.75)
parser.add_argument("--threshold", "-t", type=float, default=6)
parser.add_argument("--max_query", "-q", type=int, default=500)
parser.add_argument("--model", "-m", type=str, default='resnet')
parser.add_argument("--bd", dest='bd', action='store_true', help='let boundary detection work')
parser.add_argument("--bl", dest='bl', action='store_true', help='let blacklight work')
args = parser.parse_args()

LOGGER.info(f"loading model")
if args.model == "resnet":
    model = models.resnet50(pretrained=True).cuda()
    index = np.genfromtxt('utils/clean_resnet.csv', delimiter=',').astype(np.int32)
    n, k, epsilon_p = 128, 16, 0.052  # refer to Eq. (1)
elif args.model == "mobilenet":
    model = models.mobilenet_v2(pretrained=True).cuda()
    index = np.genfromtxt('utils/clean_mobilenet.csv', delimiter=',').astype(np.int32)
    n, k, epsilon_p = 128, 16, 0.039  # refer to Eq. (1)
else:
    raise ValueError("Invalid model")
model.eval()

LOGGER.info(f"loading data")
num_clean, num_reference = 500, args.max_query
mean = torch.tensor([0.485, 0.456, 0.406]).reshape([3, 1, 1])
std = torch.tensor([0.229, 0.224, 0.225]).reshape([3, 1, 1])
np.random.seed(66)
clean_index = index[np.floor(np.linspace(0, index.shape[0] - 1, num_clean)).astype(np.int32)]  # select clean examples
reference_index = np.setdiff1d(np.arange(50000), clean_index)  # remove clean examples
reference_index = np.random.choice(reference_index, num_reference, replace=False).astype(np.int32)  # select reference examples
ds = datasets.ImageFolder(args.v, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ]))
img_clean, fimg_clean, img_reference, fimg_reference, label = [], [], [], [], []
for i in clean_index:
    img_clean.append(ds[i][0])
    fimg_clean.append(np.fft.fftshift(np.fft.fft2(ds[i][0].numpy())))
    label.append(ds[i][1])
for i in reference_index:
    img_reference.append(ds[i][0])
    fimg_reference.append(np.fft.fftshift(np.fft.fft2(ds[i][0].numpy())))


LOGGER.info("attack")
for i in range(num_clean):
    img, fimg = img_clean[i], fimg_clean[i]
    r_l_left, r_l_right = 0, args.radius
    # calculate r_l via binary search in [0, r_h]
    while r_l_right - r_l_left > 1e-3:
        r_l_mid = (r_l_left + r_l_right) / 2
        bp_filter = band_pass_filter(r_l_mid, args.radius)
        fpass_clean = fimg * bp_filter
        pass_clean = np.fft.ifft2(np.fft.ifftshift(fpass_clean)).real.astype(np.float32)
        pass_clean = torch.tensor(pass_clean)
        if torch.norm(pass_clean) < args.threshold * 0.7:
            r_l_right = r_l_mid
        else:
            r_l_left = r_l_mid
    
    r_l, r_h = r_l_right, args.radius
    if r_l + 1e-6 > r_h:  # r_l == r_h
        LOGGER.info(f"{i}, Fail")
        continue
    bp_filter = band_pass_filter(r_l, r_h)
    fpass_clean = fimg * bp_filter
    pass_clean = np.fft.ifft2(np.fft.ifftshift(fpass_clean)).real.astype(np.float32)
    pass_clean = torch.tensor(pass_clean)

    perturb = torch.zeros([num_reference] + list(img.shape))
    for j in range(num_reference):
        fpass_reference_j = fimg_reference[j] * bp_filter
        pass_reference_j = np.fft.ifft2(np.fft.ifftshift(fpass_reference_j)).real.astype(np.float32)
        pass_reference_j = torch.tensor(pass_reference_j)
        perturb[j] = calculate_perturb(pass_clean, pass_reference_j, args.threshold)

    adv_img = torch.clamp(img + perturb, min=0, max=1)
    preds, ambiguities = norm_model(adv_img)

    # attack succeeds
    if torch.sum(preds != label[i]) == 0:
        LOGGER.info(f"{i}, Fail")

    # attack fails
    else:
        index_adv = torch.nonzero(preds != label[i])[0][0].item()  # first successful adversarial example
        if args.bl:
            log_file = f'fattack_bl_{args.model}_{args.threshold}.csv'
            adv_img = adv_img[: index_adv + 1]
            detection_query = blacklight(adv_img)
            if detection_query < 0:
                LOGGER.info(f"{i}, Succeed, {index_adv + 1} queries")
                write_log(log_file, [i, index_adv + 1, torch.norm(adv_img[index_adv] - img).item()])
            else:
                LOGGER.info(f"{i}, Detected, {detection_query} queries")

        elif args.bd:
            log_file = f'fattack_bd_{args.model}_{args.threshold}.csv'
            ambiguities = ambiguities[: index_adv + 1].numpy()
            detection_query = boundary_detection(ambiguities, n, k, epsilon_p)
            if detection_query < 0:
                LOGGER.info(f"{i}, Succeed, {index_adv + 1} queries")
                write_log(log_file, [i, index_adv + 1, torch.norm(adv_img[index_adv] - img).item()])
            else:
                LOGGER.info(f"{i}, Detected, {detection_query} queries")

        else:
            log_file = f'fattack_{args.model}_{args.threshold}.csv'
            LOGGER.info(f"{i}, Succeed, {index_adv + 1} queries")
            write_log(log_file, [i, index_adv + 1, torch.norm(adv_img[index_adv] - img).item()])
