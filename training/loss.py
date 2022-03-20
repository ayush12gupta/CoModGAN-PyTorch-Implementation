# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as T
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from .utils import resize_img
os.environ['TORCH_HOME'] = '/shared/storage/cs/staffstore/ag2157/pretrained/'

#----------------------------------------------------------------------------
# class VGGPerceptualLoss(torch.nn.Module):
#     def __init__(self, resize=True):
#         super(VGGPerceptualLoss, self).__init__()
#         blocks = []
#         blocks.append(torchvision.models.vgg16(pretrained=True).cuda().features[:4].eval())
#         blocks.append(torchvision.models.vgg16(pretrained=True).cuda().features[4:9].eval())
#         blocks.append(torchvision.models.vgg16(pretrained=True).cuda().features[9:16].eval())
#         blocks.append(torchvision.models.vgg16(pretrained=True).cuda().features[16:23].eval())
#         for bl in blocks:
#             for p in bl.parameters():
#                 p.requires_grad = False
#         self.blocks = torch.nn.ModuleList(blocks)
#         self.transform = torch.nn.functional.interpolate
#         self.resize = resize
#         self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1))
#         self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1))

#     def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
#         if input.shape[1] != 3:
#             input = input.repeat(1, 3, 1, 1)
#             target = target.repeat(1, 3, 1, 1)
#         input = (input-self.mean) / self.std
#         target = (target-self.mean) / self.std
#         if self.resize:
#             input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
#             target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
#         loss = 0.0
#         x = input
#         y = target
#         for i, block in enumerate(self.blocks):
#             x = block(x)
#             y = block(y)
#             if i in feature_layers:
#                 loss += torch.nn.functional.l1_loss(x, y)
#             if i in style_layers:
#                 act_x = x.reshape(x.shape[0], x.shape[1], -1)
#                 act_y = y.reshape(y.shape[0], y.shape[1], -1)
#                 gram_x = act_x @ act_x.permute(0, 2, 1)
#                 gram_y = act_y @ act_y.permute(0, 2, 1)
#                 loss += torch.nn.functional.l1_loss(gram_x, gram_y)
#         return loss
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


class Vgg16(torch.nn.Module):
    def __init__(self, content_wt=1e0, style_wt = 1e5):
        super(Vgg16, self).__init__()
        features = torchvision.models.vgg16(pretrained=True).features
        self.to_relu_1_2 = torch.nn.Sequential()
        self.to_relu_2_2 = torch.nn.Sequential()
        self.to_relu_3_3 = torch.nn.Sequential()
        self.to_relu_4_3 = torch.nn.Sequential()
        self.loss_mse = torch.nn.MSELoss()

        self.CONTENT_WEIGHT = content_wt
        self.STYLE_WEIGHT = style_wt

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def get_outputs(self, x, device):
        self.to_relu_1_2 = self.to_relu_1_2.to(device)
        self.to_relu_2_2 = self.to_relu_2_2.to(device)
        self.to_relu_3_3 = self.to_relu_3_3.to(device)
        self.to_relu_4_3 = self.to_relu_4_3.to(device)
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out

    def forward(self, y, y_hat):
        device = y_hat.get_device()
        # print(device, y_hat.)
        assert y_hat.get_device() is device
        # y = y.cpu()
        # y_hat = y_hat.cpu()
        # aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0

        y_features = self.get_outputs(y, device)
        y_hat_features = self.get_outputs(y_hat, device)

        # Calculating content loss
        recon = y_features[1]
        recon_hat = y_hat_features[1]
        content_loss = self.CONTENT_WEIGHT * self.loss_mse(recon_hat, recon)
        aggregate_content_loss += content_loss.data*10
        return aggregate_content_loss#.to(device)


#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, ldmks, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, fitting, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.fitting = fitting
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.vgg_loss = Vgg16() #VGGPerceptualLoss()
        self.pl_mean = torch.zeros([], device=device)

    def run_G(self, z, c, img, mask, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(img, mask, ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits
    
    def gen_img(self, img, texture):
        n_b = img.size()[0]
        imgen = resize_img(img, 224)     # Resizing the image to 224x224 for the 3dmm encoder
        shape = self.fitting.forward(imgen)
        textures = Textures(verts_uvs=self.fitting.facemodel.verts_uvs.repeat(n_b, 1, 1), faces_uvs=self.fitting.facemodel.face_buf.repeat(n_b, 1, 1), maps=texture)
        meshes = Meshes(shape, self.fitting.facemodel.face_buf.repeat(n_b, 1, 1), textures)
        rendered_img = self.fitting.renderer(meshes)
        rendered_img = resize_img(rendered_img, 512)   # Resizing back to 512x512 for computing losses
        return rendered_img[..., :3], rendered_img[..., 3:]

    def accumulate_gradients(self, phase, real_img, real_c, mask, gen_z, gen_c, ldmks, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_imageq = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        
        l1_weight = 70
        sym_weight = 70
        loss_l1 = loss_vgg = loss_Dgen = loss_Gmain = loss_Dreal = loss_sym = None
        if do_imageq:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, real_img, mask, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                loss_vgg = self.vgg_loss(real_img, gen_img)#*5
                gen_img_mirr = torch.fliplr(gen_img)
                loss_sym = abs(torch.nn.functional.l1_loss(gen_img, gen_img_mirr))*sym_weight
                # training_stats.report('Loss/scores/fake', gen_logits)
                # training_stats.report('Loss/signs/fake', gen_logits.sign())
                rend_img, rend_mask = self.gen_img(real_img, gen_img)
                loss_l1 = abs(torch.nn.functional.l1_loss(rend_img*rend_mask, real_img*rend_mask))*l1_weight
                training_stats.report('Loss/G/L1_loss', loss_l1)
                # training_stats.report('Loss/G/Perceptual', loss_vgg)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                if loss_vgg is None:
                    (loss_l1+loss_sym).mean().mul(gain).backward()
                else:
                    (loss_l1+loss_vgg+loss_sym).mean().mul(gain).backward()

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, real_img, mask, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                rend_img, rend_mask = self.gen_img(real_img, gen_img)
                loss_l1 = abs(torch.nn.functional.l1_loss(rend_img*rend_mask, real_img*rend_mask))*l1_weight
                training_stats.report('Loss/G/loss', loss_Gmain)
                training_stats.report('Loss/G/L1loss', loss_l1)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain + loss_l1*0).mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], real_img[:batch_size], mask[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                # print(gen_img.size(), real_img.size())
                # loss_l1 = abs(torch.nn.functional.l1_loss(gen_img, real_img[:batch_size]))*l1_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        if do_Dmain:
            loss_Dgen = 0
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, real_img, mask, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                
                if do_Dmain:
                    loss_Dreal = 0
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
 
                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                if do_Dmain and do_Dr1:
                    (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()
                elif do_Dr1:
                    (real_logits * 0 + loss_Dr1).mean().mul(gain).backward()
                else:
                    (real_logits * 0 + loss_Dreal).mean().mul(gain).backward()

        if loss_l1 is None:
            loss_l1 = torch.Tensor([0]).cuda()
        if loss_vgg is None:
            loss_vgg = torch.Tensor([0]).cuda()
        if loss_Gmain is None:
            loss_Gmain = torch.Tensor([0]).cuda()
        if loss_Dgen is None:
            loss_Dgen = torch.Tensor([0]).cuda()
        if loss_Dreal is None:
            loss_Dreal = torch.Tensor([0]).cuda()
        if loss_sym is None:
            loss_sym = torch.Tensor([0]).cuda()
        # print(loss_l1.mean())
        # print(loss_vgg.mean())
        # print(loss_Gmain.mean())
        # print(loss_Dgen.mean())
        # print(loss_Dreal)
        # print(loss_Dreal.mean())
        return loss_l1.mean(), loss_vgg.mean(), loss_Gmain.mean(), loss_Dgen.mean(), loss_Dreal.mean(), loss_sym.mean()

#----------------------------------------------------------------------------
