
import glob
from tqdm import tqdm
from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.submissions import get_cfg as get_submission_cfg
from core.utils.misc import process_cfg
from core.FlowFormer import build_flowformer
from scipy import interpolate
import math
import itertools
import imageio
import sys
sys.path.append('core')
# from utils import flow_viz
# import datasets

# from attr import validate


# from utils.utils import InputPadder, forward_interpolate
# from utils import frame_utils


# from raft import RAFT


TRAIN_SIZE = [480, 800]


# class InputPadder:
#     """ Pads images such that dimensions are divisible by 8 """

#     def __init__(self, dims, mode='sintel'):
#         self.ht, self.wd = dims[-2:]
#         pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
#         pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
#         if mode == 'sintel':
#             self._pad = [pad_wd//2, pad_wd - pad_wd //
#                          2, pad_ht//2, pad_ht - pad_ht//2]
#         elif mode == 'kitti432':
#             self._pad = [0, 0, 0, 432 - self.ht]
#         elif mode == 'kitti400':
#             self._pad = [0, 0, 0, 400 - self.ht]
#         elif mode == 'kitti376':
#             self._pad = [0, 0, 0, 376 - self.ht]
#         else:
#             self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

#     def pad(self, *inputs):
#         return [F.pad(x, self._pad, mode='constant', value=0.0) for x in inputs]

#     def unpad(self, x):
#         ht, wd = x.shape[-2:]
#         c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
#         return x[..., c[0]:c[1], c[2]:c[3]]


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]


def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(
        patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(
            weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd //
                         2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='constant', value=0.0) for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(
        ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


@torch.no_grad()
def create_sintel_submission(model, output_path='sintel_submission_multi8_768', sigma=0.05):
    """ Create submission for the Sintel leaderboard """
    print("no warm start")
    # print(f"output path: {output_path}")
    IMAGE_SIZE = [436, 1024]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    model.eval()
    for dstype in ['final', "clean"]:
        test_dataset = datasets.MpiSintel_submission(
            split='test', aug_params=None, dstype=dstype, root="./dataset/Sintel/test")
        epe_list = []
        for test_id in range(len(test_dataset)):
            if (test_id+1) % 100 == 0:
                print(f"{test_id} / {len(test_dataset)}")
                # break
            image1, image2, (sequence, frame) = test_dataset[test_id]
            image1, image2 = image1[None].cuda(), image2[None].cuda()

            flows = 0
            flow_count = 0

            for idx, (h, w) in enumerate(hws):
                image1_tile = image1
                image2_tile = image2
                flow_pre, flow_low = model(image1_tile, image2_tile)

                padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1],
                           h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
                flows += F.pad(flow_pre * weights[idx], padding)
                flow_count += F.pad(weights[idx], padding)

            flow_pre = flows / flow_count
            flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)


@torch.no_grad()
def create_kitti_submission(model, output_path='kitti_submission', sigma=0.05):
    """ Create submission for the Sintel leaderboard """

    IMAGE_SIZE = [432, 1242]

    print(f"output path: {output_path}")
    print(f"image size: {IMAGE_SIZE}")
    print(f"training size: {TRAIN_SIZE}")

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, (432, 1242), TRAIN_SIZE, sigma)
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        new_shape = image1.shape[1:]
        if new_shape[1] != IMAGE_SIZE[1]:   # fix the height=432, adaptive ajust the width
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[0] = 432
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        # padding the image to height of 432
        padder = InputPadder(image1.shape, mode='kitti432')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            flow_pre, _ = model(image1_tile, image2_tile)

            padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1],
                       h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = padder.unpad(flow_pre[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

        flow_img = flow_viz.flow_to_image(flow)
        image = Image.fromarray(flow_img)
        if not os.path.exists(f'vis_kitti_3patch'):
            os.makedirs(f'vis_kitti_3patch/flow')
            os.makedirs(f'vis_kitti_3patch/image')

        image.save(f'vis_kitti_3patch/flow/{test_id}.png')
        imageio.imwrite(
            f'vis_kitti_3patch/image/{test_id}_0.png', image1[0].cpu().permute(1, 2, 0).numpy())
        imageio.imwrite(
            f'vis_kitti_3patch/image/{test_id}_1.png', image2[0].cpu().permute(1, 2, 0).numpy())


@torch.no_grad()
def validate_kitti(model, sigma=0.05):
    IMAGE_SIZE = [376, 1242]
    TRAIN_SIZE = [376, 720]

    hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        new_shape = image1.shape[1:]
        if new_shape[1] != IMAGE_SIZE[1]:
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[0] = 376
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        padder = InputPadder(image1.shape, mode='kitti376')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            flow_pre, flow_low = model(image1_tile, image2_tile)

            padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1],
                       h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = padder.unpad(flow_pre[0]).cpu()
        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def validate_sintel(model, sigma=0.05):
    """ Peform validation using the Sintel (train) split """

    IMAGE_SIZE = [436, 1024]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    model.eval()
    results = {}
    for dstype in ['final', "clean"]:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)

        epe_list = []

        for val_id in range(len(val_dataset)):
            if val_id % 50 == 0:
                print(val_id)

            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            flows = 0
            flow_count = 0

            for idx, (h, w) in enumerate(hws):
                image1_tile = image1[:, :, h:h +
                                     TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
                image2_tile = image2[:, :, h:h +
                                     TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]

                flow_pre, _ = model(image1_tile, image2_tile, flow_init=None)

                padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1],
                           h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
                flows += F.pad(flow_pre * weights[idx], padding)
                flow_count += F.pad(weights[idx], padding)

            flow_pre = flows / flow_count
            flow_pre = flow_pre[0].cpu()

            epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" %
              (dstype, epe, px1, px3, px5))
        results[f"{dstype}_tile"] = np.mean(epe_list)

    return results


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to('cuda')


@torch.no_grad()
def eval_flow(args, model, val_dataset=None, sigma=0.05, index=None):
    DEVICE = 'cuda'
    slide_len = 10
    # model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load(args.model))
    # model = model.module
    # model.to(DEVICE)
    model.eval()
    IMAGE_SIZE = [480, 854]
    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
    out_list, epe_list = [], []
    data_dir = args.data_dir
    print('computing all pairwise optical flows for {}...'.format(data_dir))
    flow_out_dir = os.path.join(data_dir, 'flow_exhaustive')
    os.makedirs(flow_out_dir, exist_ok=True)
    img_files = sorted(glob.glob(os.path.join(data_dir, 'color', '*.jpg')))
    num_imgs = len(img_files)
    pbar = tqdm(total=num_imgs)
    print('num_imgs = ', num_imgs)
    i = index
    for j in range(i - 1, i-slide_len, -1):
        imfile1 = img_files[i]
        imfile2 = img_files[j]
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)
        new_shape = image1[0].shape[1:]
        # print(image1.shape)
        if new_shape[1] != IMAGE_SIZE[1]:
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[0] = 480
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
        # print(image1.shape)
        padder = InputPadder(image1.shape, mode='DAVIS')
        image1, image2 = padder.pad(
            image1.cuda(), image2.cuda())
        # print(image1.shape)
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0],
                                 w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h +
                                 TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            # print(image1_tile.shape)
            flow_pre, flow_low = model(image1_tile, image2_tile)

            padding = (w+1, IMAGE_SIZE[1]-w-TRAIN_SIZE[1]+1,
                       h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)
        # print(flows.shape)
        flow_pre = flows / flow_count
        flow = padder.unpad(flow_pre[0]).cpu()
        flow_up_np = flow.squeeze().permute(1, 2, 0).cpu().numpy()
        # print(flow_up_np.shape)
        save_file = os.path.join(flow_out_dir,
                                 '{}_{}.npy'.format(os.path.basename(imfile1), os.path.basename(imfile2)))
        np.save(save_file, flow_up_np)
    #     pbar.update(1)
    # pbar.close()
    # epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
    # mag = torch.sum(flow_gt**2, dim=0).sqrt()

    # epe = epe.view(-1)
    # mag = mag.view(-1)
    # val = valid_gt.view(-1) >= 0.5

    # out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    # epe_list.append(epe[val].mean().item())
    # out_list.append(out[val].cpu().numpy())

    # epe_list = np.array(epe_list)
    # out_list = np.concatenate(out_list)

    # epe = np.mean(epe_list)
    # f1 = 100 * np.mean(out_list)

    # print("Validation KITTI: %f, %f" % (epe, f1))
    # return {'kitti-epe': epe, 'kitti-f1': f1}

    for j in range(i-slide_len, i):
        imfile1 = img_files[j]
        imfile2 = img_files[i]
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)
        new_shape = image1[0].shape[1:]
        if new_shape[1] != IMAGE_SIZE[1]:
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[0] = 480
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
        # print(image1.shape)
        padder = InputPadder(image1.shape, mode='DAVIS')
        image1, image2 = padder.pad(
            image1.cuda(), image2.cuda())
        # print(image1.shape)
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0],
                                 w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h +
                                 TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]

            flow_pre, flow_low = model(image1_tile, image2_tile)

            padding = (w+1, IMAGE_SIZE[1]-w-TRAIN_SIZE[1]+1,
                       h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = padder.unpad(flow_pre[0]).cpu()
        flow_up_np = flow.squeeze().permute(1, 2, 0).cpu().numpy()
        save_file = os.path.join(flow_out_dir,
                                 '{}_{}.npy'.format(os.path.basename(imfile1), os.path.basename(imfile2)))
        np.save(save_file, flow_up_np)

    # epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
    # mag = torch.sum(flow_gt**2, dim=0).sqrt()

    # epe = epe.view(-1)
    # mag = mag.view(-1)
    # val = valid_gt.view(-1) >= 0.5

    # out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    # epe_list.append(epe[val].mean().item())
    # out_list.append(out[val].cpu().numpy())

    # epe_list = np.array(epe_list)
    # out_list = np.concatenate(out_list)

    # epe = np.mean(epe_list)
    # f1 = 100 * np.mean(out_list)

    # print("Validation KITTI: %f, %f" % (epe, f1))
    # return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='load model')
    parser.add_argument('--eval', help='eval benchmark')
    parser.add_argument('--data_dir', help='data directory')
    parser.add_argument('--small', action='store_true', help='use small model')
    args = parser.parse_args()

    exp_func = None
    cfg = get_submission_cfg()
    # if args.eval == 'sintel_submission':
    #     exp_func = create_sintel_submission
    #     cfg = get_submission_cfg()
    # elif args.eval == 'kitti_submission':
    #     exp_func = create_kitti_submission
    #     cfg = get_submission_cfg()
    # elif args.eval == 'sintel_validation':
    #     exp_func = validate_sintel
    #     cfg = get_submission_cfg()
    # elif args.eval == 'kitti_validation':
    #     exp_func = validate_kitti
    #     cfg = get_submission_cfg()
    # else:
    #     print(f"EROOR: {args.eval} is not valid")
    cfg.update(vars(args))

    print(cfg)
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load('./checkpoint/kitti.pth'))

    model.cuda()
    model.eval()

    eval_flow(args, model.module)
