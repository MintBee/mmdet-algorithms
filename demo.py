# TODO should optimize for detecto-rs
from mmdet.apis import init_detector, inference_detector
import argparse
import os
import glob
from pathlib import Path

file_path = os.path.abspath(__file__)
root_dir = Path(file_path).parent.absolute()
checkpoint_dir = os.path.join(root_dir, 'checkpoints')
config_dir = os.path.join(root_dir, "configs")

algorithms = ['detecto-rs', 'mask2former', 'solo', 'solov2', 'yolact']

config_files = {
    'detecto-rs' : os.path.join(config_dir, 'detectors/detectors_htc_r101_20e_coco.py'),
    'mask2former': os.path.join(config_dir, 'mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'),
    'solo': os.path.join(config_dir, 'solo/decoupled_solo_light_r50_fpn_3x_coco.py'),
    'solov2': os.path.join(config_dir, 'solov2/solov2_x101_dcn_fpn_3x_coco.py'),
    'yolact': os.path.join(config_dir, 'yolact/yolact_r101_1x8_coco.py')
}

checkpoint_files = {
    'detecto-rs': os.path.join(checkpoint_dir, 'detectors/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth'),
    'mask2former': os.path.join(checkpoint_dir, 'mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth'),
    'solo': os.path.join(checkpoint_dir, 'solo/decoupled_solo_r50_fpn_3x_coco_20210821_042504-7b3301ec.pth'),
    'solov2': os.path.join(checkpoint_dir, 'solov2/solov2_x101_dcn_fpn_3x_coco_20220513_214337-aef41095.pth'),
    'yolact': os.path.join(checkpoint_dir, 'yolact/yolact_r101_1x8_coco_20200908-4cbe9101.pth')
}


def main():
    args = get_args()
    selected_alg = args.algorithm
    config_file = config_files[selected_alg]
    checkpoint_file = checkpoint_files[selected_alg]

    vrlab_path = '/usr/local/vrlab'
    # TODO Make input and output path based on arguments
    dataset_path = os.path.join(vrlab_path, 'datasets/custom')
    output_path = os.path.join(vrlab_path, os.path.join('results', selected_alg))

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
        model.show_result(img_path, result, score_thr=0.25, out_file=out_file_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm',
                        choices=algorithms,
                        help='Algorithm to activate')
    # TODO Add arguments for input & output path
    return parser.parse_args()


if __name__ == '__main__':
    main()