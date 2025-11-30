import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        """
        ArcFace: Additive Angular Margin Loss for Deep Face Recognition
        Args:
            in_features: Size of each input sample (embedding dimension)
            out_features: Number of classes (identities)
            s: Norm of input feature (scale factor)
            m: Margin
        """
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # Weight parameter (centers for each class)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # Thresholds to ensure numerical stability
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # 1. Normalize Features and Weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # 2. Calculate Sine from Cosine
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        # 3. Calculate Phi (cos(theta + m)) using trig identities
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # 4. Handle numerical stability conditions
        if self.m > 0.0:
            # If cosine > th, use phi. Else use a Taylor expansion approximation 
            # to prevent gradients from exploding or vanishing
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 5. Create One-Hot encoding to apply margin only to the ground truth class
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # 6. Apply margin: Use phi for correct class, cosine for others
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 7. Scale
        output *= self.s
        
        return output

# Example of how you might add CosFace in the future
class CosFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super(CosFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output