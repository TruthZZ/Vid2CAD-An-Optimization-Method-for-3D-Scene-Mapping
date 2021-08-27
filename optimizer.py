from posix import listdir
import torch
import torch.nn as nn
from torch.nn.modules.module import register_module_backward_hook
from losses import OverallLoss
import os
import json
import torch.optim as optim
from torch.autograd import Variable
from scipy.spatial.transform import Rotation as sRota

class optimizer():
    def __init__(self):
        super(optimizer, self).__init__()

        self.frame_width = 960
        self.frame_height = 720

        self.big_epoch_num = 5000
        self.lr = 0.002
        self.loss_func = OverallLoss(self.frame_width, self.frame_height)

        path_frame = './Syn_data/frames/'
        path_jsons = './Syn_data/jittered/'
        listdir_frame = os.listdir(path_frame)
        listdir_jsons = os.listdir(path_jsons)
        self.objects_frame_dict = {}
        self.frame_num = len(listdir_frame)

        self.all_frame_dict = {}
        for i in range(self.frame_num):
            tmp_path = path_jsons + listdir_jsons[i]
            with open(tmp_path, 'r') as f:
                frame_dict = json.load(f)
            self.all_frame_dict[i] = frame_dict

            frame_objects = frame_dict['objects'].keys()
            for keys in frame_objects:
                if keys not in self.objects_frame_dict.keys():
                    self.objects_frame_dict[keys] = []
                obj_info_dict = frame_dict['objects'][keys]
                pixel_coors = []
                pixel_coors += obj_info_dict['center']
                pixel_coors += obj_info_dict['bounding_box_2d']
                flag = True
                for pc in pixel_coors:
                    if pc < 0:
                        flag = False
                if pixel_coors[0] > (self.frame_width-1) or pixel_coors[2] > (self.frame_width-1) or pixel_coors[3] > (self.frame_width-1):
                    flag = False
                if pixel_coors[1] > (self.frame_height-1) or (self.frame_height-1) or (self.frame_height-1):
                    flag = False
                if flag:
                    self.objects_frame_dict[keys].append(i)

        # self.res_list = []
        # self.all_loss_list = []
        self.overall_loss = {}

    def optm_single_obj(self, obj_name):
        self.overall_loss[obj_name] = []
        frame_list = self.objects_frame_dict[obj_name]
        t = Variable(torch.randn((3, 1)), requires_grad = True)
        R = Variable(torch.randn((4)), requires_grad = True)
        s = Variable(torch.rand(3, 1), requires_grad = True)
        parameters_list = [t, R, s]

        frame_auxiliary_dict = {}
        for idx in frame_list:
            temp_dict = {}
            temp_dict['kx'] = Variable(torch.rand(1), requires_grad=True)
            temp_dict['ky'] = Variable(torch.rand(1), requires_grad=True)
            temp_dict['beta'] = Variable(torch.rand(1), requires_grad=True)
            frame_auxiliary_dict[idx] = temp_dict
            parameters_list.append(frame_auxiliary_dict[idx]['kx'])
            parameters_list.append(frame_auxiliary_dict[idx]['ky'])
            parameters_list.append(frame_auxiliary_dict[idx]['beta'])

        optimizer_1 = optim.Adam(parameters_list, lr=self.lr)

        for big_epoch in range(self.big_epoch_num):
            optimizer_1.zero_grad()
            loss = torch.tensor([0.], requires_grad=True)
            loss_by_frame = []
            for frame_idx in frame_list:
                frame_dict = self.all_frame_dict[frame_idx]
                K = Variable(torch.tensor(frame_dict['K']))
                et = Variable(torch.tensor(frame_dict['et']))
                ER = Variable(torch.tensor(frame_dict['ER']))

                kx = frame_auxiliary_dict[frame_idx]['kx']
                ky = frame_auxiliary_dict[frame_idx]['ky']
                beta = frame_auxiliary_dict[frame_idx]['beta']

                obj_info_dict = frame_dict['objects'][obj_name]
                bbox_3d = obj_info_dict['bounding_box_3d']
                center_x = obj_info_dict['center'][0] / self.frame_width
                center_y = obj_info_dict['center'][1] / self.frame_height
                center = Variable(torch.tensor([center_x, center_y]))
                bbox_x_min = obj_info_dict['bounding_box_2d'][0] / self.frame_width
                bbox_x_max = obj_info_dict['bounding_box_2d'][1] / self.frame_width
                bbox_y_min = obj_info_dict['bounding_box_2d'][2] / self.frame_height
                bbox_y_max = obj_info_dict['bounding_box_2d'][3] / self.frame_height
                bbox_2d = Variable(torch.tensor([bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max]))
                Rotation = Variable(torch.tensor(obj_info_dict['Rotation']))
                s_cls = Variable(torch.tensor(obj_info_dict['origin_scale']))

                loss_list = self.loss_func(Rotation, center, bbox_2d,
                                    bbox_3d,
                                    ER, et, K,
                                    kx, ky, beta,
                                    t, R, s,
                                    s_cls)
                loss_by_frame.append([loss_list[0].item(), loss_list[1].item(), loss_list[2].item(), loss_list[3].item(), loss_list[4].item(), loss_list[5].item()])
                loss = loss + loss_list[0]

            # self.all_loss_list.append(loss_by_frame)

            loss /= len(frame_list)
            self.overall_loss[obj_name].append(loss.item())
            loss.backward()
            optimizer_1.step()

            if big_epoch % 50 == 0:
                # temp_res_dict = {}
                # temp_res_dict['t'] = t.tolist()
                # temp_res_dict['R'] = R.tolist()
                # temp_res_dict['s'] = s.tolist()
                # self.res_list.append(temp_res_dict)
                print("Optimizing Object:", obj_name, "Step:", big_epoch+1, " Loss:", loss.item())

            # if big_epoch == 0 or i == 2:
            #     self.lr /= 2

        res_dict = {}
        res_dict['t'] = t.tolist()
        R = R / torch.norm(R)
        R = R.tolist()
        R_scipy = sRota.from_quat(R)
        R = R_scipy.as_matrix()
        res_dict['R'] = R.tolist()
        res_dict['s'] = s.tolist()

        return res_dict

    def optm_all_obj(self):
        obj_scene_dict = {}
        for name in self.objects_frame_dict.keys():
            res_dict = self.optm_single_obj(name)
            obj_scene_dict[name] = res_dict

        return obj_scene_dict

if __name__ == '__main__':
    test_optimizer = optimizer()
    # res_dict = test_optimizer.optm_single_obj('chair_10')
    # print(res_dict)
    scene_dict = test_optimizer.optm_all_obj()
    # with open('./all_res.json', 'w') as f:
    #     json.dump(test_optimizer.res_list, f, indent=4, sort_keys=False)
    # with open('./all_loss_list.json', 'w') as f:
    #     json.dump(test_optimizer.all_loss_list, f, indent=4, sort_keys=False)
    with open('./scene_objects.json', 'w') as f:
        json.dump(scene_dict, f, indent=4, sort_keys=False)
    with open('./overall_loss.json', 'w') as f:
        json.dump(test_optimizer.overall_loss, f, indent=4, sort_keys=False)
