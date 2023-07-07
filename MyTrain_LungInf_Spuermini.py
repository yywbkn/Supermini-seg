
import torch
from torch.autograd import Variable
import os,random
import argparse
from datetime import datetime
from Code.utils.dataloader_LungInf import get_loader
from Code.utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


def dice_loss(mask,pred,ep=1e-8):
    intersection = 2 * torch.sum(pred * mask) + ep
    union = torch.sum(pred) + torch.sum(mask) + ep
    loss = 1 - intersection / union
    return loss



def joint_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (0.2*wbce + 0.8*wiou ).mean()



def train(train_loader, model, optimizer, epoch, train_save):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.6,1]
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, edges = pack
            images = Variable(images).cuda()

            gts = Variable(gts).cuda()
            edges = Variable(edges).cuda()

            # ---- rescaling the inputs (img/gt/edge) ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                edges = F.upsample(edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # ---- forward ----
            # lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
            lateral_map_2, lateral_map_3, lateral_map_4, lateral_map_5, edge01, edge02 = model(images)

            # ---- loss function ----
            loss5 = joint_loss(lateral_map_5, gts)
            loss4 = joint_loss(lateral_map_4, gts)
            loss3 = joint_loss(lateral_map_3, gts)
            loss2 = joint_loss(lateral_map_2, gts)
            loss1 = torch.nn.BCEWithLogitsLoss()(edge02, edges)
            loss0 = torch.nn.BCEWithLogitsLoss()(edge01, edges)
            # loss6 = torch.nn.BCEWithLogitsLoss()(lateral_map_2, edges)

            # loss = 12*loss0 + 12*loss1 + 5*loss2 + loss3 + loss4 + loss5 + 5*loss6
            loss = 2 * loss0 + 2 * loss1 + 15 * loss2 + 3 *loss3 + 2 * loss4 + loss5
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
        # ---- train logging ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [lateral-edge: {:.4f}, '
                  'lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record1.show(),
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
    # ---- save model_lung_infection ----
    save_path = './Snapshots/save_weights/{}/'.format(train_save)
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % 1 == 0:
        torch.save(model.state_dict(), save_path + 'SuperMini-Seg-%d.pth' % (epoch+1))
        print('[Saving Snapshot:]', save_path + 'SuperMini-Seg-%d.pth' % (epoch+1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # hyper-parameters
    parser.add_argument('--epoch', type=int, default=100,help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4,help='learning rate')
    parser.add_argument('--classes', type=int, default=1, help='No. of classes in the dataset')
    parser.add_argument('--batchsize', type=int, default=2,help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352,help='set the size of training sample')
    parser.add_argument('--clip', type=float, default=0.5,help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50,help='every n epochs decay learning rate')
    parser.add_argument('--is_thop', type=bool, default=False,help='whether calculate FLOPs/Params (Thop)')
    parser.add_argument('--gpu_device', type=int, default=0,help='choose which GPU device you want to use')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers in dataloader. In windows, set num_workers=0')
    # model_lung_infection parameters
    parser.add_argument('--net_channel', type=int, default=32,help='internal channel numbers in the SuperMini-Seg, default=32, try larger for better accuracy')
    parser.add_argument('--n_classes', type=int, default=1,help='binary segmentation when n_classes=1')
    parser.add_argument('--backbone', type=str, default='SuperMini-Seg',help='change different backbone, choice: VGGNet16, ResNet50, Res2Net50')
    # training dataset
    parser.add_argument('--train_path', type=str,default='./Dataset/TrainingSet/LungInfection-Train/Doctor-label')
    #parser.add_argument('--train_path', type=str, default='./Dataset/TrainingSet/LungInfection-Train/all-label02')
    parser.add_argument('--train_save', type=str, default='SuperMini-Seg',help='If you use custom save path, please edit `--is_semi=True` and `--is_pseudo=True`')
    parser.add_argument('--seed', type=int, default=1)
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        cudnn.deterministic = True


    # # ---- build models ----
    torch.cuda.set_device(opt.gpu_device)


    from models import SuperMini as net
    model = net.Super_MiniSeg(opt.classes, aux=True)
    model = model.cuda()


    # ---- calculate FLOPs and Params ----
    if opt.is_thop:
        from Code.utils.utils import CalParams
        x = torch.randn(1, 3, opt.trainsize, opt.trainsize).cuda()
        CalParams(model, x)

    # ---- load training sub-modules ----
    BCE = torch.nn.BCEWithLogitsLoss()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root,batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=opt.num_workers)
    total_step = len(train_loader)

    # ---- start !! -----
    print("#"*20, "\nStart Training (SuperMini-Seg-{})\n{}\nThis code is written for 'SpuerMini-Seg:', 2022.\n"
                  "----\nPlease cite the paper if you use this code and dataset. "
                  "And any questions feel free to contact me ".format(opt.backbone, opt), "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt.train_save)
