import torch
import torch.nn as nn

class OverallLoss(nn.Module):
    def __init__(self, frame_width, frame_height):
        super(OverallLoss, self).__init__()

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.pixel_center_loss = Pixel_Center_Loss()
        self.world_center_loss = World_Center_Loss(self.frame_width, self.frame_height)
        self.rotation_loss = Rotation_Loss()
        self.scale_bbox_loss = Scale_Bbox_Loss()
        self.scale_class_loss = Scale_Class_Loss()

        self.coefficients = [0.2, 0.2, 0.2, 0.2, 0.2]

    def forward(self, Rotation, center, bbox, bbox_3d, ER, et, K, kx, ky, beta, t, R, s, s_cls):
        overall_loss = torch.tensor([0], dtype=torch.float32)

        k = torch.cat((kx, ky))

        R = R / torch.norm(R)
        R_mat = [[],[],[]]
        R_mat[0].append((1-2*(torch.square(R[1])+torch.square(R[2]))).unsqueeze(0))
        R_mat[0].append((2*(R[0]*R[1]-R[2]*R[3])).unsqueeze(0))
        R_mat[0].append((2*(R[0]*R[2]+R[1]*R[3])).unsqueeze(0))
        R_mat[1].append((2*(R[0]*R[1]+R[2]*R[3])).unsqueeze(0))
        R_mat[1].append((1-2*(torch.square(R[0])+torch.square(R[2]))).unsqueeze(0))
        R_mat[1].append((2*(R[1]*R[2]-R[0]*R[3])).unsqueeze(0))
        R_mat[2].append((2*(R[0]*R[2]-R[1]*R[3])).unsqueeze(0))
        R_mat[2].append((2*(R[1]*R[2]+R[0]*R[3])).unsqueeze(0))
        R_mat[2].append((1-2*(torch.square(R[0])+torch.square(R[1]))).unsqueeze(0))

        R_mat[0] = torch.cat(R_mat[0]).unsqueeze(0)
        R_mat[1] = torch.cat(R_mat[1]).unsqueeze(0)
        R_mat[2] = torch.cat(R_mat[2]).unsqueeze(0)
        R_mat = torch.cat(R_mat, dim=0).float()

        extrinsics = torch.cat([ER, et.unsqueeze(0)], dim=0)
        temp = torch.tensor([0,0,0,1]).unsqueeze(0).t()
        extrinsics = torch.cat([extrinsics, temp], dim=1)
        extrinsics_inv = torch.linalg.inv(extrinsics)
        bbox_3d_corners = [
            torch.tensor([bbox_3d[0], bbox_3d[2], bbox_3d[4]]),
            torch.tensor([bbox_3d[0], bbox_3d[2], bbox_3d[5]]),
            torch.tensor([bbox_3d[0], bbox_3d[3], bbox_3d[4]]),
            torch.tensor([bbox_3d[0], bbox_3d[3], bbox_3d[5]]),
            torch.tensor([bbox_3d[1], bbox_3d[2], bbox_3d[4]]),
            torch.tensor([bbox_3d[1], bbox_3d[2], bbox_3d[5]]),
            torch.tensor([bbox_3d[1], bbox_3d[3], bbox_3d[4]]),
            torch.tensor([bbox_3d[1], bbox_3d[3], bbox_3d[5]])
        ]
        for i in range(8):
            bbox_3d_corners[i] = torch.mul(s.t(), bbox_3d_corners[i])
            bbox_3d_corners[i] = bbox_3d_corners[i].t().squeeze()
            bbox_3d_corners[i] = torch.mv(R_mat, bbox_3d_corners[i]).unsqueeze(0).t()
            bbox_3d_corners[i] += t
        for i in range(8):
            bbox_3d_corners[i] = torch.cat([bbox_3d_corners[i], torch.tensor([1]).unsqueeze(0)])
            bbox_3d_corners[i] = torch.mm(bbox_3d_corners[i].t(), extrinsics_inv)
            bbox_3d_corners[i] = bbox_3d_corners[i].t()[:3]
            bbox_3d_corners[i] = bbox_3d_corners[i] / bbox_3d_corners[i][2]
            bbox_3d_corners[i] = torch.mm(K, bbox_3d_corners[i]).t()
        bbox_3d_corners = torch.cat(bbox_3d_corners)
        mins = torch.min(bbox_3d_corners, dim=0)
        maxs = torch.max(bbox_3d_corners, dim=0)
        bbox_2d = torch.cat([mins[0][0].unsqueeze(0)/self.frame_width,
                            maxs[0][0].unsqueeze(0)/self.frame_width,
                            mins[0][1].unsqueeze(0)/self.frame_height,
                            maxs[0][1].unsqueeze(0)/self.frame_height])

        pixel_center_loss = self.pixel_center_loss(k, center)
        world_center_loss = self.world_center_loss(kx, ky, beta, ER, et, K, t)
        rotation_loss = self.rotation_loss(Rotation, ER, R_mat)
        scale_bbox_loss = self.scale_bbox_loss(bbox_2d, bbox)
        scale_cls_loss = self.scale_class_loss(s, s_cls)

        # print("pixel_center_loss:", pixel_center_loss)
        # print("world_center_loss:", world_center_loss)
        # print("rotation_loss:", rotation_loss)
        # print("scale_bbox_loss:", scale_bbox_loss)
        # print("scale_cls_loss:", scale_cls_loss)

        overall_loss += self.coefficients[0] * pixel_center_loss
        overall_loss += self.coefficients[1] * world_center_loss
        overall_loss += self.coefficients[2] * rotation_loss
        overall_loss += self.coefficients[3] * scale_bbox_loss
        overall_loss += self.coefficients[4] * scale_cls_loss

        return [overall_loss, pixel_center_loss, world_center_loss, rotation_loss, scale_bbox_loss, scale_cls_loss]

class Pixel_Center_Loss(nn.Module):
    def __init__(self):
        super(Pixel_Center_Loss, self).__init__()
        self.l1loss = nn.L1Loss(reduction='mean')

    def forward(self, k, c):

        return self.l1loss(k, c)

class World_Center_Loss(nn.Module):
    def __init__(self, frame_width, frame_height):
        super(World_Center_Loss, self).__init__()
        self.l1loss = nn.L1Loss(reduction='mean')

        self.frame_width = frame_width
        self.frame_height = frame_height

    def forward(self, kx, ky, beta, ER, et, K, t):
        # camera_coors = torch.tensor([[beta*kx*960, beta*ky*720, beta]], requires_grad=True).t()
        camera_coors = torch.cat([beta*kx*self.frame_width, beta*ky*self.frame_height, beta], dim=0).unsqueeze(0).t()
        # print(beta, kx, ky)
        # print(camera_coors)
        camera_coors = torch.mm(torch.linalg.inv(K), camera_coors)
        l2wMat = torch.cat([ER, et.unsqueeze(0)], dim=0)
        temp = torch.tensor([0,0,0,1]).unsqueeze(0).t()
        l2wMat = torch.cat([l2wMat, temp], dim=1)
        camera_coors = torch.cat([camera_coors, torch.tensor([1]).unsqueeze(0)])
        world_coors = torch.mm(camera_coors.t(), l2wMat).squeeze()[:3]
        # print(world_coors, t.t().squeeze())
        loss = self.l1loss(world_coors, t.t().squeeze())

        return loss

class Rotation_Loss(nn.Module):
    def __init__(self):
        super(Rotation_Loss, self).__init__()
        self.l2loss = nn.MSELoss(reduction='mean')

    def forward(self, Ri, ER, R_mat):
        loss1 = self.l2loss(Ri, torch.mm(ER, R_mat))
        
        return loss1

class Scale_Bbox_Loss(nn.Module):
    def __init__(self):
        super(Scale_Bbox_Loss, self).__init__()
        self.l1loss = nn.L1Loss(reduction='mean')

    def forward(self, bbox_v, bbox_i):
        
        return self.l1loss(bbox_v, bbox_i)

class Scale_Class_Loss(nn.Module):
    def __init__(self):
        super(Scale_Class_Loss, self).__init__()
        self.l1loss = nn.L1Loss(reduction='mean')

    def forward(self, scale_v, scale_c):

        return self.l1loss(scale_v.squeeze(), scale_c)