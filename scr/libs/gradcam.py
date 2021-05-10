import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import os
device = torch.device('cuda:0'
  if torch.cuda.is_available() else 'cpu')

import urllib
import pickle
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
import cv2

class GradCAM:
  def __init__(self, model, convlayer, in_img):
    self.model = model
    self.fmap = None # 特徴マップ
    self.grad = None # 勾配
    self.y = None # ネットワークの出力
    self.handles = [] # hookした関数のハンドル

    def _store_fmap(module, x_in, x_out):
      self.fmap = x_out # 畳み込み層の出力
    def _store_grad(module, g_in, g_out):
      self.grad = g_out[0] # 勾配
    h1 = convlayer.register_forward_hook(_store_fmap)
    h2 = convlayer.register_backward_hook(_store_grad)
    self.handles = [h1, h2]

    self.model.eval() # 推論モードへ
    self.y = self.model(in_img)
    _, pred = torch.max(self.y, 1)
    self.pred = pred[0]

  def compute(self, t_class):
    yc = F.one_hot(torch.tensor([t_class]),
            num_classes=self.y.shape[-1]).float()
    self.model.zero_grad()
    self.y.backward(yc.to(device), retain_graph=True)
    print('shape:', self.fmap.shape)

    grad_np = self.grad.detach().cpu().numpy()[0]
    fmap_np = self.fmap.detach().cpu().numpy()[0]
    weights = np.mean(grad_np, axis=(1, 2))
    gcam = np.dot(fmap_np.transpose(1, 2, 0), weights)
    gcam = np.maximum(gcam, 0) # ReLUを取る
    gcam = gcam / np.max(gcam) # 0-1へ正規化
    print('Grad-CAM を計算しました')
    return gcam

  def clear_hooks(self):
    for h in self.handles[:]: # スライスでリスト複製
      h.remove()
      self.handles.remove(h)



def save_heatmap(filepath, gcam, img):
  img = img[0].permute(1, 2, 0).cpu()
  img = Image.fromarray(np.uint8(img*255))
  gcam_hi = np.uint8(
    Image.fromarray(np.uint8(gcam * 255)).resize((32, 32), Image.BILINEAR) )/255
  hmap_np = cm.get_cmap('jet')(gcam_hi)
  hmap = Image.fromarray(np.uint8(hmap_np*255))
  hmap_on = Image.blend(img.convert('RGBA'), hmap, 0.4)
  # hmap_on.save(filepath)
  print(f'Grad-CAM の結果を保存しました:¥n{filepath}')
  return hmap_on

def concat_imshow(img1, img2,filepath):
  img = Image.new('RGB', (img1.width + img2.width, img1.height))
  img.paste(img1, (0, 0))
  img.paste(img2, (img1.width, 0))
  plt.imshow(np.array(img))
  # plt.show()
  plt.savefig(filepath)
  plt.close()

# def GradCAM(net, img, save_path):

#     net.eval()

#     def __extract(grad):
#         global feature_grad
#         feature_grad = grad

#     # get features from the last convolutional layer
#     x = net.conv1(img)
#     x = net.bn1(x)
#     # x = net.relu(x)
#     # x = net.maxpool(x)
#     x = net.layer1(x)
#     x = net.layer2(x)
#     x = net.layer3(x)
#     x = net.layer4(x)
#     features = x

#     # hook for the gradients
#     def __extract_grad(grad):
#         global feature_grad
#         feature_grad = grad
#     if features.requires_grad:
#       features.register_hook(__extract_grad)

#     # get the output from the whole VGG architecture
#     x = F.avg_pool2d(x, 4)
#     x = x.view(x.size(0), -1)
#     output = net.linear(x)
#     pred = torch.argmax(output).item()
#     print(pred)

#     # get the gradient of the output
#     output[:, pred].backward()

#     # pool the gradients across the channels
#     pooled_grad = torch.mean(feature_grad, dim=[0, 2, 3])

#     # weight the channels with the corresponding gradients
#     # (L_Grad-CAM = alpha * A)
#     features = features.detach()
#     for i in range(features.shape[1]):
#         features[:, i, :, :] *= pooled_grad[i] 

#     # average the channels and create an heatmap
#     # ReLU(L_Grad-CAM)
#     heatmap = torch.mean(features, dim=1).squeeze()
#     heatmap = heatmap.cpu()
#     heatmap = np.maximum(heatmap, 0)

#     # normalization for plotting
#     heatmap = heatmap / torch.max(heatmap)
#     heatmap = heatmap.detach().numpy()

#     # project heatmap onto the input image
#     # img = cv2.imread(img_fpath)
#     aa=img[0,:,:,:]
#     heatmap = cv2.resize(heatmap, (aa.shape[1], aa.shape[2]))
#     # heatmap = np.uint8(255 * heatmap)
#     aa = aa.permute(1, 2, 0)
#     aa = 0.5 * aa + 0.50
#     # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     # superimposed_img = heatmap * 0.4 + img
#     superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))
#     superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
#     plt.imshow(superimposed_img)
#     plt.save(superimposed_img)
#     plt.close()
#     # plt.show()