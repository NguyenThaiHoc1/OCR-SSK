import cv2
import numpy as np
import math
import traceback
import copy
import time
from argparse import Namespace

from ocr_package.tools.infer.utility import get_rotate_crop_image
import ocr_package.tools.infer.utility as utility

from ocr_package.ppocr.postprocess import build_post_process
from ocr_package.ppocr.utils.utility import get_image_file_list, check_and_read
from ocr_package.ppocr.utils.logging import get_logger

logger = get_logger()


class TextRecognizer(object):
    def __init__(self, args):

        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm

        args.rec_char_dict_path = args.rec_char_dict_path
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_dict_path": args.rec_char_dict_path,
            "use_space_char": args.use_space_char
        }

        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'rec', logger)
        self.benchmark = args.benchmark
        self.use_onnx = args.use_onnx

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape

        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        if self.use_onnx:
            w = self.input_tensor.shape[3:][0]
            if isinstance(w, str):
                pass
            elif w is not None and w > 0:
                imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        st = time.time()

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []

            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            # max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            input_dict = {}
            input_dict[self.input_tensor.name] = norm_img_batch
            outputs = self.predictor.run(self.output_tensors,
                                         input_dict)
            preds = outputs[0]

            rec_result = self.postprocess_op(preds)

            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        return rec_res, time.time() - st


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_recognizer = TextRecognizer(args)
    valid_image_file_list = []
    img_list = []

    for image_file in image_file_list:
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        ori_im = img.copy()

        height, width, _ = img.shape

        # Define dt_boxes using np.float32
        dt_boxes = [np.array([[0., 0.],
                              [width, 0.],
                              [width, height],
                              [0., height]], dtype=np.float32)]

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_list.append(img_crop)
        valid_image_file_list.append(image_file)
    try:
        rec_res, time_pred = text_recognizer(img_list)

    except Exception as E:
        logger.info(traceback.format_exc())
        logger.info(E)
        exit()
    for ino in range(len(img_list)):
        logger.info("Predicts of {}".format(valid_image_file_list[ino]))
        logger.info('Result :{}'.format(rec_res[ino]))
        logger.info('Time to predict:{}'.format(time_pred))

        print(rec_res)
        print("====")
    if args.benchmark:
        text_recognizer.autolog.report()


# Define the parameter values
para_values = {
    'use_gpu': False,
    'image_dir': './image_dir/text-rec/mg_crop_0.jpg',
    'rec_algorithm': 'SVTR_LCNet',
    'rec_model_dir': './models/text-rec/ja/model.onnx',
    'rec_image_inverse': True,
    'rec_image_shape': '3,48,-1',
    'rec_batch_num': 6,
    'max_text_length': 32,
    'rec_char_dict_path': './ocr_package/ppocr/utils/dict/japan_dict.txt',
    'use_space_char': True,
    'drop_score': 0.5,
    'benchmark': False,
    'use_onnx': True
}
para = Namespace(**para_values)

if __name__ == "__main__":
    main(para)
