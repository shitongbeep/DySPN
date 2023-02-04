from torchvision.transforms import transforms
import os
import os.path
import glob
import numpy as np
from PIL import Image
import torch.utils.data.dataset as data
import torch
from dataloaders.bottomcrop import BottomCrop

input_options = ['d', 'rgb', 'rgbd', 'g', 'gd']
tensor = transforms.ToTensor()
pil = transforms.ToPILImage()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_calib():
    """
    返回校正后的相机内参矩阵
    投影矩阵,用于从矫正后的0号相机坐标系投影到2号相机的图像平面。之发生了平移
    [1  0  0  x][fx 0  cx 0]  -> [fx 0  cx x]
    [0  1  0  y][0  fy cy 0]  -> [0  fy cy y]
    [0  0  1  z][0  0  1  0]  -> [0  0  1  z]
    [0  0  0  1][0  0  0  1]  -> [0  0  0  1]
    """
    calib = open("dataloaders/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]), (3, 4)).astype(np.float32)
    K = Proj[:3, :3]  # camera matrix

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    # K[0, 2] = K[0, 2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    # K[1, 2] = K[1, 2] - 11.5  # from width = 375 to 352, with a 11.5-pixel cut on both sides
    K[0, 2] = K[0, 2] - 13
    K[1, 2] = K[1, 2] - 27.5 * 2
    return K


def train_transform(rgb, sparse, target, args):
    seed = np.random.randint(int(args['train_params']['seed']))

    oheight = args['dataset_params']['image_h']
    owidth = args['dataset_params']['image_w']

    transforms_list = [
        # transforms.Rotate(angle),
        # transforms.Resize(s),
        BottomCrop((oheight, owidth)),
        transforms.RandomHorizontalFlip(0.5)
    ]

    transform_geometric = transforms.Compose(transforms_list)

    jitter = args['dataset_params']['jitter']
    if sparse is not None:
        set_seed(seed)
        sparse = transform_geometric(tensor(sparse))
    if target is not None:
        set_seed(seed)
        target = transform_geometric(tensor(target))
    if rgb is not None:
        set_seed(seed)
        rgb = transform_geometric(tensor(rgb))
        transform_rgb = transforms.Compose([transforms.ColorJitter(jitter, jitter, jitter, 0)])
        rgb = transform_rgb(rgb)

    # random crop
    # if small_training == True:
    if args['dataset_params']['random_crop'] is True:
        crop_transform = transforms.Compose(
            [transforms.RandomCrop((args['dataset_params']['random_crop_height'], args['dataset_params']['random_crop_width']))])
        if rgb is not None:
            set_seed(seed)
            rgb = crop_transform(rgb)
        if sparse is not None:
            set_seed(seed)
            sparse = crop_transform(sparse)
        if target is not None:
            set_seed(seed)
            target = crop_transform(target)

    return rgb, sparse, target


def val_transform(rgb, sparse, target, args):

    oheight = args['dataset_params']['image_h']
    owidth = args['dataset_params']['image_w']

    transform = transforms.Compose([
        BottomCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform(tensor(rgb))
    if sparse is not None:
        sparse = transform(tensor(sparse))
    if target is not None:
        target = transform(tensor(target))

    return rgb, sparse, target


def no_transform(rgb, sparse, target, args):
    if rgb is not None:
        rgb = tensor(rgb)
    if sparse is not None:
        sparse = tensor(sparse)
    if target is not None:
        target = tensor(target)
    return rgb, sparse, target


def get_paths_and_transform(mode, args):
    use_rgb = True
    use_d = True
    data_folder = args['dataset_params']['data_folder']
    data_folder_rgb = args['dataset_params']['data_folder_rgb']
    if mode == "train":
        use_gt = True
        transform = train_transform
        glob_d = os.path.join(data_folder, 'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png')
        glob_gt = os.path.join(data_folder, 'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png')

        # glob_gt = os.path.join(data_folder, 'data_depth_velodyne/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png')

        def get_rgb_paths(p):
            ps = p.split('/')
            date_liststr = []
            date_liststr.append(ps[-5][:10])
            # kitti raw 的路径
            pnew = '/'.join(date_liststr + ps[-5:-4] + ps[-2:-1] + ['data'] + ps[-1:])
            pnew = os.path.join(data_folder_rgb, pnew)
            return pnew
    elif mode == "val":
        use_gt = True
        if args['train_params']['val'] == "full":
            transform = val_transform
            glob_d = os.path.join(data_folder, 'data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png')
            glob_gt = os.path.join(data_folder, 'data_depth_annotated/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png')

            # glob_gt = os.path.join(data_folder, 'data_depth_velodyne/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png')

            def get_rgb_paths(p):
                ps = p.split('/')
                date_liststr = []
                date_liststr.append(ps[-5][:10])
                pnew = '/'.join(date_liststr + ps[-5:-4] + ps[-2:-1] + ['data'] + ps[-1:])
                pnew = os.path.join(data_folder_rgb, pnew)
                return pnew

        elif args['train_params']['val'] == "select":
            # transform = no_transform
            transform = val_transform
            glob_d = os.path.join(data_folder, "data_depth_selection/val_selection_cropped/velodyne_raw/*.png")
            glob_gt = os.path.join(data_folder, "data_depth_selection/val_selection_cropped/groundtruth_depth/*.png")

            def get_rgb_paths(p):
                return p.replace("groundtruth_depth", "image")
        else:
            raise (RuntimeError("Unrecognized validation mode"))
    elif mode == "test":
        use_gt = False
        transform = no_transform
        glob_d = os.path.join(data_folder, "data_depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png")
        glob_gt = None  # "test_depth_completion_anonymous/"
        glob_rgb = os.path.join(data_folder, "data_depth_selection/test_depth_completion_anonymous/image/*.png")
    else:
        raise ValueError("Unrecognized mode " + str(mode))

    if glob_gt is not None:
        # train or val-full or val-select
        # glob.glob() Return a list of paths matching a pathname pattern
        paths_d = sorted(glob.glob(glob_d))
        paths_gt = sorted(glob.glob(glob_gt))
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]
    else:
        # test only has d or rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_gt = [None] * len(paths_rgb)
        if mode == "test_prediction":
            paths_d = [None] * len(paths_rgb)  # test_prediction has no sparse depth
        else:
            paths_d = sorted(glob.glob(glob_d))

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_d) == 0 and use_d:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0 and use_rgb:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_gt) == 0 and use_gt:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        print(len(paths_rgb), len(paths_d), len(paths_gt))
        raise (RuntimeError("Produced different sizes for datasets"))
    paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    return paths, transform


def rgb_read(filename):
    '''只是转化成了numpy array'''
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png


def depth_read(filename):
    '''加载稀疏深度图 --> numpy [n,1]'''
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)

    depth = depth_png.astype(np.float16) / 256.
    depth = np.expand_dims(depth, -1)
    return depth


img_to_tensor = transforms.ToTensor()


class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset
    """

    def __init__(self, mode, args):
        self.args = args
        self.mode = mode
        paths, transform = get_paths_and_transform(mode, args)
        self.paths = paths
        self.transform = transform
        self.K = load_calib()
        self.threshold_translation = 0.1

    def __getraw__(self, index):
        rgb = None
        if self.args['model_params']['mode'] != 'test':
            rgb = rgb_read(self.paths['rgb'][index]) if (self.paths['rgb'][index] is not None) else None
        sparse = depth_read(self.paths['d'][index]) if (self.paths['d'][index] is not None) else None
        target = depth_read(self.paths['gt'][index]) if self.paths['gt'][index] is not None else None
        return rgb, sparse, target

    def __getitem__(self, index):
        rgb, sparse, target = self.__getraw__(index)
        rgb, sparse, target = self.transform(rgb, sparse, target, self.args)
        candidates = {"rgb": rgb, "d": sparse, "gt": target, 'K': torch.from_numpy(self.K)}

        items = {key: val.float() for key, val in candidates.items() if val is not None}
        return items

    def __len__(self):
        return len(self.paths['gt'])
