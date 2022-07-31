# TODO should optimize for detecto-rs
from mmdet.apis import init_detector, inference_detector
import mmcv
import argparse
import os
import glob
from pathlib import Path

home_dir = str(Path.home())
mmdet_dir = os.path.join(home_dir, 'utilities/mmdetection')
config_dir = os.path.join(mmdet_dir, "configs")


config_files = {
    ""
}


def main():
    args = get_args()
    config_file = args.config_file
    checkpoint_file = args.checkpoint

    vrlab_path = '/usr/local/vrlab'
    dataset_path = os.path.join(vrlab_path, 'datasets/custom')
    output_path = os.path.join(vrlab_path, 'results/solo')

    try:
        os.mkdir(output_path)
    except OSError as error:
        print(error)

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test a single image
    img_paths = glob.glob(os.path.join(dataset_path, '*.png'))
    for img_path in img_paths:
        result = inference_detector(model, img_path)
        out_file_path = os.path.join(output_path, os.path.basename(img_path))
        print(out_file_path)
        model.show_result_ins(img_path, result, model.CLASSES, score_thr=0.25, out_file=out_file_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file',
                        default='../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py',
                        metavar='FILE',
                        help='path to config file')
    parser.add_argument('--checkpoint',
                        default='../checkpoints/DECOUPLED_SOLO_R50_3x.pth',
                        metavar='FILE',
                        help='path to checkpoint(pretrained) file')
    return parser.parse_args()


if __name__ == '__main__':
    main()