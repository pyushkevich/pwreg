import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def_device = torch.device("cpu")

class Image2D:

    def __init__(self, T=None, ras=None):
        self.T = T
        self.ras = ras

    @staticmethod
    def load(fname, dtype=torch.short, device=def_device):

        img = sitk.ReadImage(fname)

        # Get the tensor representing the image, unsqueeze for Torch to have dims 1x1xWxH
        t = torch.tensor(sitk.GetArrayFromImage(img), dtype=dtype, device=device).squeeze()
        t = torch.unsqueeze(torch.unsqueeze(t, 0), 0)

        # Construct the NIFTI matrix for the image
        idir, ispc, iorg = img.GetDirection(), img.GetSpacing(), img.GetOrigin()
        idir = np.array(idir).reshape(2, 2) if len(idir) == 4 else np.array(idir).reshape(3, 3)[0:2, 0:2]
        ras = np.eye(3)
        ras[0:2, 0:2] = idir @ np.array([[-ispc[0], 0.], [0., -ispc[1]]])
        ras[2, 0:2] = np.array(iorg[0:2])

        return Image2D(t, ras)

    def downsample(self, factor):
        x = Image2D()
        x.T = F.interpolate(self.T, scale_factor=factor, align_corners=False, mode='bilinear', recompute_scale_factor=True)
        rassc = np.eye(3)
        rassc[0:2, 0:2] = np.diag(np.flip(np.array(self.T.shape[-2:]) / np.array(x.T.shape[-2:])))
        x.ras = rassc @ self.ras
        x.ras[0:3, 2] = x.ras[2, 0:3] + ((x.ras - self.ras) @ np.array([0.5, 0.5, 0.]).reshape(3, 1)).squeeze()
        return x


class Dataset:

    def __init__(self, img_fix, img_mov, img_mask, dtype=torch.short):

        # Read the intensity images
        self.I_fix = Image2D.load(img_fix, torch.float)
        self.I_mov = Image2D.load(img_mov, torch.float)

        # Read the chunk mask image
        M_chunk = Image2D.load(img_mask, torch.long)
        T = torch.nn.functional.one_hot(M_chunk.T)[:, :, :, :, 1:].transpose(0, 4).squeeze(4).type(torch.float)
        self.M_fix = Image2D(T, M_chunk.ras)


class RigidProblem(nn.Module):

    def __init__(self, I_fix, I_mov, M_fix):

        # Standard initialization
        super(RigidProblem, self).__init__()

        # Assign the data
        self.I_fix, self.I_mov, self.M_fix = I_fix, I_mov, M_fix

        # Fix the number of masks, i.e., batch size
        self.k = M_fix.T.shape[0]

        # Generate affine transformation from pytorch grid into RAS coords for fixed image
        h, w = I_fix.T.shape[-2:]
        U = np.array([[0, w / 2., w / 2. - .5], [h / 2., 0, h / 2. - .5], [0., 0., 1.]])
        self.R_fix = torch.tensor(I_fix.ras @ U).type(torch.float)
        self.Q_fix = torch.tensor(np.linalg.inv(self.R_fix.numpy()))

        # Generate same for the moving image
        h, w = I_mov.T.shape[-2:]
        U = np.array([[0, w / 2., w / 2. - .5], [h / 2., 0, h / 2. - .5], [0., 0., 1.]])
        self.R_mov = torch.tensor(I_mov.ras @ U).type(torch.float)
        self.Q_mov = torch.tensor(np.linalg.inv(self.R_mov.numpy()))

    def forward(self, theta, dx, dy):

        # Generate the rigid matrices from the inputs
        M = torch.stack((
                torch.stack((torch.cos(theta), -torch.sin(theta), dx)),
                torch.stack((torch.sin(theta), torch.cos(theta), dy)),
                torch.stack((torch.zeros(8), torch.zeros(8), torch.ones(8)))
            )).transpose(0, 2).transpose(1, 2)

        A = self.Q_mov @ M @ self.R_fix

        # Generate a sampling grid for this matrix
        grid = F.affine_grid(A[:, 0:2, :], self.M_fix.T.shape, align_corners=False)

        # Apply sampling grid to moving image
        resampled = F.grid_sample(self.I_mov.T.repeat(self.k,1,1,1), grid, align_corners=False)
        return resampled, M, A, grid









