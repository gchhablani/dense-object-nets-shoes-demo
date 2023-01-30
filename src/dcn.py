import torch
import torch.nn as nn

class DenseCorrespondenceNetwork(nn.Module):
    def __init__(self, fcn, descriptor_dimension, image_width=640,
                 image_height=480, normalize=False):
        """
        :param fcn:
        :type fcn:
        :param descriptor_dimension:
        :type descriptor_dimension:
        :param image_width:
        :type image_width:
        :param image_height:
        :type image_height:
        :param normalize: If True normalizes the feature vectors to lie on unit ball
        :type normalize:
        """

        super(DenseCorrespondenceNetwork, self).__init__()

        self._fcn = fcn
        self._descriptor_dimension = descriptor_dimension
        self._image_width = image_width
        self._image_height = image_height
        self._normalize = normalize

    def forward(self, img_tensor):
        """
        Simple forward pass on the network.
        Does NOT normalize the image
        D = descriptor dimension
        N = batch size
        :param img_tensor: input tensor img.shape = [N, D, H , W] where
                    N is the batch size
        :type img_tensor: torch.Variable or torch.Tensor
        :return: torch.Variable with shape [N, D, H, W],
        :rtype:
        """

        res = self._fcn(img_tensor)
        if self._normalize:
            #print "normalizing descriptor norm"
            norm = torch.norm(res, 2, 1) # [N,1,H,W]
            res = res/norm


        return res
    
    def forward_on_img_tensor(self, img):
        """
        Deprecated, use `forward` instead
        Runs the network forward on an img_tensor
        :param img: (C x H X W) in range [0.0, 1.0]
        :return:
        """

        img = img.unsqueeze(0)
        img = torch.tensor(img, device=torch.device("cuda"))
        res = self._fcn(img)
        res = res.squeeze(0)
        res = res.permute(1, 2, 0)
        res = res.data.cpu().numpy().squeeze()

        return res