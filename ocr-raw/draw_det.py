import os
import cv2
import ast

from argparse import Namespace
from ocr_package.tools.infer import utility

para_values = {
    'folder_det_input': './image_dir/text-det',
    'folder_det_output': './output_dir/text-det',
    'information_file': './output_dir/text-det/det_results.txt',

}

"""
    cách xử lý bbox
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    [
        [181.0, 742.0], 
        [485.0, 750.0], 
        [484.0, 782.0], 
        [180.0, 774.0]
    ]

"""


def main(args):
    with open(para_values['information_file']) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        extracted_info = line.split('\t')
        filename = extracted_info[0]
        filename_out = 'drawed_' + filename
        bbox = extracted_info[1]
        bbox = ast.literal_eval(bbox)

        path_inp = os.path.join(args.folder_det_input, filename)
        path_out = os.path.join(args.folder_det_output, filename_out)
        img_np = cv2.imread(path_inp)

        drawed_img_np = utility.draw_text_det_res(
            dt_boxes=bbox, img=img_np
        )
        cv2.imwrite(path_out, drawed_img_np)


if __name__ == "__main__":
    para = Namespace(**para_values)
    main(para)
