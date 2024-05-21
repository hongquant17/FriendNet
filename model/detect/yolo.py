import torch
import torch.nn as nn

from model.detect.backbone import Backbone, Multi_Concat_Block, Conv
from torchvision import transforms
from utils.utils_bbox import DecodeBox
from utils.utils_detect import (cvtColor, get_anchors, get_classes, preprocess_input, resize_image)


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(13, 9, 5)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv3 = Conv(4 * c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.cv3(torch.cat([m(x1) for m in self.m] + [x1], 1))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        'model_path': 'logs/ep300-loss0.038-val_loss0.040.pth',
        'classes_path': 'data/voc_classes.txt',
        # ---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        # ---------------------------------------------------------------------#
        'anchors_path': 'data/yolo_anchors.txt',
        'anchors_mask': [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        # ---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        # ---------------------------------------------------------------------#
        'input_shape': [256, 256],
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        'confidence': 0.5,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        'nms_iou': 0.3,
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        'letterbox_image': True,
        'cuda': True,
    }
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return 'Unrecognized attribute name '' + n + '''
    
    def __init__(self, anchors_mask, num_classes, pretrained=False, return_backbone_feature=False):
        super(YoloBody, self).__init__()
        # -----------------------------------------------#
        #   定义了不同yolov7-tiny的参数
        # -----------------------------------------------#
        transition_channels = 16
        block_channels = 16
        panet_channels = 16
        e = 1
        n = 2
        ids = [-1, -2, -3, -4]
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        # -----------------------------------------------#

        # ---------------------------------------------------#
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80, 80, 512
        #   40, 40, 1024
        #   20, 20, 1024
        # ---------------------------------------------------#
        self.backbone = Backbone(transition_channels, block_channels, n, pretrained=pretrained)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 16)
        self.conv_for_P5 = Conv(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2 = Conv(transition_channels * 16, transition_channels * 8)
        self.conv3_for_upsample1 = Multi_Concat_Block(transition_channels * 16, panet_channels * 4,
                                                      transition_channels * 8, e=e, n=n, ids=ids)

        self.conv_for_P4 = Conv(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1 = Conv(transition_channels * 8, transition_channels * 4)
        self.conv3_for_upsample2 = Multi_Concat_Block(transition_channels * 8, panet_channels * 2,
                                                      transition_channels * 4, e=e, n=n, ids=ids)

        self.down_sample1 = Conv(transition_channels * 4, transition_channels * 8, k=3, s=2)
        self.conv3_for_downsample1 = Multi_Concat_Block(transition_channels * 16, panet_channels * 4,
                                                        transition_channels * 8, e=e, n=n, ids=ids)

        self.down_sample2 = Conv(transition_channels * 8, transition_channels * 16, k=3, s=2)
        self.conv3_for_downsample2 = Multi_Concat_Block(transition_channels * 32, panet_channels * 8,
                                                        transition_channels * 16, e=e, n=n, ids=ids)

        self.rep_conv_1 = Conv(transition_channels * 4, transition_channels * 8, 3, 1)
        self.rep_conv_2 = Conv(transition_channels * 8, transition_channels * 16, 3, 1)
        self.rep_conv_3 = Conv(transition_channels * 16, transition_channels * 32, 3, 1)

        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1).to('cuda:1')
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)
        
        self.anchors, self.num_anchors = get_anchors('data/yolo_anchors.txt')
        self.anchors_mask, self.num_classes = anchors_mask, num_classes
        self.letterbox_image = True
        self.bbox_util = DecodeBox(self.anchors, num_classes, (256, 256),
                                   anchors_mask)
        self.model_path = self.get_defaults('model_path')
        self.return_backbone_feature = return_backbone_feature

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self

    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)

        P5 = self.sppcspc(feat3)
        P5_conv = self.conv_for_P5(P5)
        P5_upsample = self.upsample(P5_conv)
        P4 = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        P4 = self.conv3_for_upsample1(P4)

        P4_conv = self.conv_for_P4(P4)
        P4_upsample = self.upsample(P4_conv)
        P3 = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        P3 = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)

        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)
        # ---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size, 75, 80, 80)
        # ---------------------------------------------------#
        out2 = self.yolo_head_P3(P3).to('cuda')
        # ---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size, 75, 40, 40)
        # ---------------------------------------------------#
        out1 = self.yolo_head_P4(P4).to('cuda')
        # ---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size, 75, 20, 20)
        # ---------------------------------------------------#
        out0 = self.yolo_head_P5(P5).to('cuda')

        # return [out0, out1, out2]
        # TODO
        if self.return_backbone_feature:
            return [out0, out1, out2], feat3
        return [out0, out1, out2]
    
    def get_detection_guidance(self, image, resize=False):
        image_shape = image.shape[2:4]
        if resize:
            resize = transforms.Resize((256, 256))
            image = resize(image)

        with torch.no_grad():
            self.net = YoloBody(self.anchors_mask, self.num_classes)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.net.load_state_dict(torch.load(self.model_path, map_location=device))
            self.net = self.net.fuse().eval()
            # print('{} model, and classes loaded.'.format(self.model_path))
            if self.cuda:
                # self.net = nn.DataParallel(self.net)
                self.net = self.net.module if isinstance(self.net, nn.DataParallel) else self.net
                self.net = self.net.to('cuda:0')
                
            outputs = self.net(image.to('cuda'))
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, (256, 256),
                                                         image_shape, self.letterbox_image, conf_thres=0.5,
                                                         nms_thres=0.3)

        batch_size = len(results)
        [h, w] = image_shape
        # Initialize the result tensor
        guidance = torch.zeros((batch_size, 3, h, w))

        for i in range(batch_size):
            if results[i] is not None:
                top_label = results[i][:, 6].to(torch.int32)
                top_conf = results[i][:, 4] * results[i][:, 5]
                top_boxes = results[i][:, :4]

                # Calculate the indices for slicing
                top, left, bottom, right = top_boxes.split(1, dim=1)
                top = torch.clamp(top.floor().to(torch.int32), min=0)
                left = torch.clamp(left.floor().to(torch.int32), min=0)
                bottom = torch.clamp(bottom.floor().to(torch.int32), max=h)
                right = torch.clamp(right.floor().to(torch.int32), max=w)

                # Update guidance tensor directly
                for index in range(len(top_label)):
                    c = top_label[index]
                    score = top_conf[index]
                    guidance[i, :, top[index]:bottom[index], left[index]:right[index]] = (c + 1) * score

        return guidance


if __name__ == '__main__':
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 5
    model = YoloBody(anchors_mask, num_classes)
    input = torch.randn((4, 3, 512, 512))
    outputs = model(input)
    for out in outputs:
        print(out.shape)
    # outputs
    # torch.Size([4, 30, 16, 16])
    # torch.Size([4, 30, 32, 32])
    # torch.Size([4, 30, 64, 64])
