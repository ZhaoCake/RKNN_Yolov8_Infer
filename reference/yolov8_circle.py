import os
import cv2
import sys
import argparse
import numpy as np
import onnxruntime as ort


OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# The follow two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 640)  # (width, height)

# Circle dataset classes based on dataset.yaml
CLASSES = ("blue_stack", "blue_space", "green_stack", "green_space", "red_stack", "red_space")

# Class colors for visualization
CLASS_COLORS = [
    (255, 0, 0),    # blue_stack - red
    (255, 255, 0),  # blue_space - cyan  
    (0, 255, 0),    # green_stack - green
    (0, 255, 255),  # green_space - yellow
    (0, 0, 255),    # red_stack - blue
    (255, 0, 255)   # red_space - magenta
]


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold."""
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def dfl(position):
    """Distribution Focal Loss (DFL) - moved to post-processing for RKNN compatibility"""
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    
    # Apply softmax
    y_exp = np.exp(y - np.max(y, axis=2, keepdims=True))
    y = y_exp / np.sum(y_exp, axis=2, keepdims=True)
    
    # Create accumulation matrix
    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1)
    y = np.sum(y * acc_metrix, axis=2)
    
    return y


def box_process(position):
    """Process box predictions with DFL"""
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy


def post_process(input_data):
    """Post-process ONNX model outputs adapted for modified YOLOv8"""
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    
    # Process each detection branch
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        # For modified model, confidence might be separate or combined
        if pair_per_branch > 2:  # If confidence is separate
            scores.append(input_data[pair_per_branch*i+2])
        else:  # Use ones like original
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints"""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescale boxes (xyxy) from img1_shape to img0_shape"""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        # ratio_pad format: (ratio, (dw, dh)) where ratio can be tuple or single value
        ratio, pad = ratio_pad
        if isinstance(ratio, tuple):
            gain = ratio[0]  # Use first ratio value
        else:
            gain = ratio

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    """Clip boxes (xyxy) to image shape (height, width)"""
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def draw(image, boxes, scores, classes):
    """Draw detection results on image"""
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        
        # Use class-specific colors
        color = CLASS_COLORS[cl % len(CLASS_COLORS)]
        cv2.rectangle(image, (top, left), (right, bottom), color, 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class ONNXModel:
    """ONNX Model wrapper"""
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
    def run(self, input_data):
        """Run inference"""
        outputs = self.session.run(self.output_names, {self.input_name: input_data[0]})
        return outputs
    
    def release(self):
        """Release resources"""
        pass


def setup_model(args):
    """Setup model based on file extension"""
    model_path = args.model_path
    if model_path.endswith('.onnx'):
        platform = 'onnx'
        model = ONNXModel(args.model_path)
    else:
        assert False, "{} is not onnx model".format(model_path)
    print('Model-{} is {} model, starting inference'.format(model_path, platform))
    return model, platform


def img_check(path):
    """Check if file is image"""
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv8 Circle Detection Inference')
    # basic params
    parser.add_argument('--model_path', type=str, required=True, help='model path, must be .onnx file')
    
    parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
    parser.add_argument('--img_save', action='store_true', default=False, help='save the result')

    # data params
    parser.add_argument('--img_folder', type=str, default='./test_images', help='img folder path')
    parser.add_argument('--img_path', type=str, default=None, help='single image path')

    args = parser.parse_args()

    # init model
    model, platform = setup_model(args)

    # Prepare image list
    img_list = []
    if args.img_path and os.path.exists(args.img_path):
        # Single image inference
        img_list = [os.path.basename(args.img_path)]
        img_folder = os.path.dirname(args.img_path)
    else:
        # Batch inference
        img_folder = args.img_folder
        if os.path.exists(img_folder):
            file_list = sorted(os.listdir(img_folder))
            for path in file_list:
                if img_check(path):
                    img_list.append(path)
        else:
            print(f"Image folder {img_folder} does not exist")
            exit(1)

    if not img_list:
        print("No images found!")
        exit(1)

    # run test
    for i in range(len(img_list)):
        print('infer {}/{}'.format(i+1, len(img_list)), end='\r')

        img_name = img_list[i]
        img_path = os.path.join(img_folder, img_name) if args.img_path is None else args.img_path
        
        if not os.path.exists(img_path):
            print("{} is not found".format(img_name))
            continue

        img_src = cv2.imread(img_path)
        if img_src is None:
            continue

        # Preprocessing
        img, ratio, (dw, dh) = letterbox(img_src, new_shape=IMG_SIZE, auto=False)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Prepare input for ONNX
        input_data = img.transpose((2, 0, 1))  # HWC to CHW
        input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
        input_data = input_data / 255.0  # Normalize to [0,1]

        # Run inference
        outputs = model.run([input_data])
        
        # Post-process results
        boxes, classes, scores = post_process(outputs)

        if args.img_show or args.img_save:
            print('\n\nIMG: {}'.format(img_name))
            img_p = img_src.copy()
            if boxes is not None:
                # Scale boxes back to original image
                boxes_scaled = scale_boxes((IMG_SIZE[1], IMG_SIZE[0]), boxes.copy(), img_src.shape[:2], (ratio, (dw, dh)))
                draw(img_p, boxes_scaled, scores, classes)

            if args.img_save:
                if not os.path.exists('./result'):
                    os.mkdir('./result')
                result_path = os.path.join('./result', img_name)
                cv2.imwrite(result_path, img_p)
                print('Detection result save to {}'.format(result_path))
                        
            if args.img_show:
                cv2.imshow("YOLOv8 Circle Detection Result", img_p)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # release
    model.release()
    print("\nInference completed!")
