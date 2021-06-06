from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8 # one cluster for each type
NUM_OBJECT_POINT = 512
g_type2class={'Car':0, 'Van':1, 'Truck':2, 'Pedestrian':3,
              'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7}
g_class2type = {g_type2class[t]:t for t in g_type2class}
g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
g_type_mean_size = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
                    'Van': np.array([5.06763659,1.9007158,2.20532825]),
                    'Truck': np.array([10.13586957,2.58549199,3.2520595]),
                    'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
                    'Person_sitting': np.array([0.80057803,0.5983815,1.27450867]),
                    'Cyclist': np.array([1.76282397,0.59706367,1.73698127]),
                    'Tram': np.array([16.17150617,2.53246914,3.53079012]),
                    'Misc': np.array([3.64300781,1.54298177,1.92320313])}
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i,:] = g_type_mean_size[g_class2type[i]]

class CenterRegressionTNet(nn.Module):
    ''' Regression network for center delta. a.k.a. T-Net.
    Input:
        object_point_cloud:  tensor in shape (B,M,C)
            point clouds in 3D mask coordinate
        one_hot_vec: tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        predicted_center:  tensor in shape (B,3)
    '''
    def __init__(self, in_dim = 4):
        super(CenterRegressionTNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_dim, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(512 + 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)




    def forward(self, point_cloud, one_hot_vec):
        batchsize = point_cloud.size()[0]
        x = point_cloud.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 512)

        x = torch.cat([one_hot_vec, x], 1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        predicted_center = self.fc3(x)

        return predicted_center



class PointNetfeat(nn.Module):
    def __init__(self, in_dim = 4, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.in_dim  = in_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat


    def forward(self, point_cloud, one_hot_vec):
        x = point_cloud.transpose(2, 1)
        n_pts = x.size()[2]

        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = F.relu(self.bn2(self.conv2(x)))
        x = pointfeat


        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = torch.cat([one_hot_vec, x], 1)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024 + 3, 1).repeat(1, 1, n_pts)
            x = torch.cat([x, pointfeat], 1)
            x = x.transpose(2, 1)
            return x


class PointNetSegmentation(nn.Module):
    ''' 3D instance segmentation PointNet v1 network.
    Input:
        point_cloud:  tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec:  tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        bn_decay:  float scalar
        end_points: dict
    Output:
        logits:  tensor in shape (B,N,2), scores for bkg/clutter and object'''
    def __init__(self, k = 2):
        super(PointNetSegmentation, self).__init__()
        self.k = k
        self.feat = PointNetfeat(global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088 + 3, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(p = 0.5)

    def forward(self, point_cloud, one_hot_vec):
        batchsize = point_cloud.size()[0]
        n_pts = point_cloud.size()[1]
        x = self.feat(point_cloud, one_hot_vec)
        x = x.transpose(2, 1)
        # size here is Bx1091xN
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout1(x)
        x = self.conv5(x)
        x = x.transpose(2,1).contiguous()
        return x


def point_cloud_masking(point_cloud, segmentation_results, end_points):
    ''' Select point cloud with predicted 3D mask,
    translate coordinates to the masked points centroid.

    Input:
        point_cloud:  tensor in shape (B,N,C)
        segmentation_results:  tensor in shape (B,N,2)
        xyz_only: boolean, if True only return XYZ channels
    Output:
        object_point_cloud:  tensor in shape (B,M,3)
            for simplicity we only keep XYZ here
            M = NUM_OBJECT_POINT as a hyper-parameter
        mask_xyz_mean:  tensor in shape (B,3)
    '''

    batch_size = point_cloud.size()[0]
    num_point = point_cloud.size()[1]
    mask_bool =(segmentation_results[:,:,0] < segmentation_results[:,:,1])
    mask = mask_bool.float()
#     print(mask.size())
    mask_count = torch.sum(mask, 1, keepdim=True).repeat(1, 1, 3)
    mask_count = mask_count.view(batch_size, 1, 3)
    point_cloud_xyz = point_cloud[:,:,:3] # only xyz
    end_points['mask'] = mask
#     print(point_cloud_xyz.size(), mask_bool.size())
    mask_xyz_mean = (mask.unsqueeze(2).repeat(1,1,3)*point_cloud_xyz).sum(dim = 1, keepdim= True)
    mask_xyz_mean = mask_xyz_mean/torch.clamp(mask_count, min = 1)
#     print(mask_xyz_mean.size())
    point_cloud_xyz_stage1 = point_cloud_xyz - mask_xyz_mean.repeat(1, num_point, 1)
#     print(point_cloud_xyz_stage1.size())

    point_cloud_stage1 = point_cloud_xyz_stage1
    npoints=NUM_OBJECT_POINT
    non_zero_indices = torch.cat([torch.where((mask>0.5))[0].unsqueeze(1), torch.where((mask>0.5))[1].unsqueeze(1)],1)
    object_pc = torch.zeros(batch_size, npoints, 3)
    for i in range(batch_size):

        all_points_indices = non_zero_indices[non_zero_indices[:,0] == i][:,1]
        if all_points_indices.size()[0] > 0:

            segmented_points = point_cloud_stage1[i][all_points_indices]
            if segmented_points.size()[0] >= npoints:
                perm = torch.randperm(segmented_points.size(0))
                object_pc[i] = segmented_points[perm[:npoints]]
            else:
                choice = np.random.choice(segmented_points.size(0),
                    npoints - segmented_points.size(0), replace=True)
                choice = np.concatenate((np.arange(segmented_points.size(0)), choice))
                np.random.shuffle(choice)
                object_pc[i] = segmented_points[choice]
    return object_pc, mask_xyz_mean.squeeze(1), end_points

class PointNet3DboxEstimationNet(nn.Module):
    ''' 3D Box Estimation PointNet v1 network.
    Input:
        object_point_cloud: TF tensor in shape (B,M,C)
            point clouds in object coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        output: TF tensor in shape (B,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
            including box centers, heading bin class scores and residuals,
            and size cluster scores and residuals
    '''
    def __init__(self, in_dim = 3):
        super(PointNet3DboxEstimationNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_dim, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(512 + 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)


        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)



    def forward(self, point_cloud, one_hot_vec):
        batchsize = point_cloud.size()[0]
        x = point_cloud.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 512)

        x = torch.cat([x, one_hot_vec], 1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.fc3(x)
        return x

def parse_output_to_tensors(output, end_points):
    ''' Parse batch output to separate tensors (added to end_points)
    Input:
        output: TF tensor in shape (B,3+2*NUM_HEADING_BIN+4*NUM_SIZE_CLUSTER)
        end_points: dict
    Output:
        end_points: dict (updated)
    '''
    output = output.cpu()
    for key in end_points:
        end_points[key] = end_points[key].cpu()

    batch_size = output.size(0)
#     center = tf.slice(output, [0,0], [-1,3])
    center = output[:,:3]

    end_points['center_boxnet'] = center

    heading_scores = output[:,3:3 + NUM_HEADING_BIN]
    heading_residuals_normalized = output[:,3 + NUM_HEADING_BIN: 3 + NUM_HEADING_BIN + NUM_HEADING_BIN]
    end_points['heading_scores'] = heading_scores # BxNUM_HEADING_BIN
    end_points['heading_residuals_normalized'] = \
        heading_residuals_normalized # BxNUM_HEADING_BIN (-1 to 1)
    end_points['heading_residuals'] = \
        heading_residuals_normalized * (np.pi/NUM_HEADING_BIN) # BxNUM_HEADING_BIN

    size_scores = output[:,3+NUM_HEADING_BIN*2:3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER] # BxNUM_SIZE_CLUSTER
    size_residuals_normalized = output[:,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER:3+NUM_HEADING_BIN*2+4*NUM_SIZE_CLUSTER]
    size_residuals_normalized = size_residuals_normalized.view(batch_size,NUM_SIZE_CLUSTER,3)# BxNUM_SIZE_CLUSTERx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['size_residuals'] = size_residuals_normalized * \
        torch.from_numpy(g_mean_size_arr).unsqueeze(0)

    return end_points


class FrustrumPointNent_v1(nn.Module):
    ''' Frustum PointNets model. The model predict 3D object masks and
    amodel bounding boxes for objects in frustum point clouds.

    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        is_training: TF boolean scalar
        bn_decay: TF float scalar
    Output:
        end_points: dict (map from name strings to tensors)'''
    def __init__(self, in_dim = 4):
        super(FrustrumPointNent_v1, self).__init__()
        self.segmentationNet = PointNetSegmentation(k = 2)
        self.centerRegressionNet = CenterRegressionTNet(in_dim = 3)
        self.amodalBoxEstimationNet = PointNet3DboxEstimationNet(in_dim = 3)


    def forward(self, point_cloud, one_hot_vec):
        end_points = dict()
        # Segmentation
        logits = self.segmentationNet(point_cloud, one_hot_vec)
        end_points['mask_logits'] = logits

        # point cloud masking according to results of object_
        object_point_cloud_xyz, mask_xyz_mean, end_points = point_cloud_masking(point_cloud, logits, end_points)
        object_point_cloud_xyz, mask_xyz_mean = object_point_cloud_xyz.cuda(), mask_xyz_mean.cuda()
        # TODO: fix it properly later
        # finding the center of the object
        center_delta = self.centerRegressionNet(object_point_cloud_xyz, one_hot_vec)
        stage1_center = center_delta + mask_xyz_mean # Bx3
        end_points['stage1_center'] = stage1_center

        # Get object point cloud in object coordinate
        object_point_cloud_xyz_new = object_point_cloud_xyz - center_delta.unsqueeze(1)

        # Amodel Box Estimation PointNet
        output = self.amodalBoxEstimationNet(object_point_cloud_xyz_new, one_hot_vec)

        end_points = parse_output_to_tensors(output, end_points)
        end_points['center'] = end_points['center_boxnet'] + stage1_center.cpu() # Bx3

        return end_points

def get_box3d_corners_helper(centers, headings, sizes):
    """ TF layer. Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    #print '-----', centers
    N = centers.size(0)
#     l = tf.slice(sizes, [0,0], [-1,1]) # (N,1)
    l = sizes[:, 0:1]
    w = sizes[:, 1:2] # (N,1)
    h = sizes[:, 2:3] # (N,1)
#     print (l.size())
    x_corners = torch.cat([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], dim=1) # (N,8)
    y_corners = torch.cat([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], dim=1) # (N,8)
    z_corners = torch.cat([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], dim=1) # (N,8)
#     print (z_corners.size())
    corners = torch.cat([x_corners.unsqueeze(1), y_corners.unsqueeze(1), z_corners.unsqueeze(1)], axis=1) # (N,3,8)
#     print (corners.size())
    c = torch.cos(headings)
    s = torch.sin(headings)
#     print (s.size())
    ones = torch.ones([N], dtype=torch.float32)
    zeros = torch.zeros([N], dtype=torch.float32)
    row1 = torch.stack([c,zeros,s], axis=1) # (N,3)
#     print(zeros.size(), row1.size())
    row2 = torch.stack([zeros,ones,zeros], axis=1)
    row3 = torch.stack([-s,zeros,c], axis=1)
    R = torch.cat([row1.unsqueeze(1), row2.unsqueeze(1), row3.unsqueeze(1)], axis=1) # (N,3,3)
#     print (row1.size(),R.size(), N)
    corners_3d = torch.matmul(R, corners) # (N,3,8)
    corners_3d += centers.unsqueeze(2).repeat(1,1,8)  # (N,3,8)
#     print(corners_3d.size())
#     tf.tile(tf.expand_dims(centers,2), [1,1,8])
    corners_3d = corners_3d.transpose(2,1) # (N,8,3)
#      tf.transpose(corners_3d, perm=[0,2,1])
    return corners_3d


def get_box3d_corners(center, heading_residuals, size_residuals):
    """ TF layer.
    Inputs:
        center: (B,3)
        heading_residuals: (B,NH)
        size_residuals: (B,NS,3)
    Outputs:
        box3d_corners: (B,NH,NS,8,3) tensor
    """
    batch_size = center.size(0)
    heading_bin_centers = torch.from_numpy(np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN)) # (NH,)
    headings = heading_residuals + heading_bin_centers.unsqueeze(0) # (B,NH)
#     print(headings.size())
    mean_sizes = torch.from_numpy(g_mean_size_arr).unsqueeze(0) + size_residuals # (B,NS,1)
    sizes = mean_sizes + size_residuals # (B,NS,3)
#     sizes = tf.tile(tf.expand_dims(sizes,1), [1,NUM_HEADING_BIN,1,1]) # (B,NH,NS,3)
    sizes = sizes.unsqueeze(1).repeat(1,NUM_HEADING_BIN,1,1)
#     print(sizes.size())
#     headings = tf.tile(tf.expand_dims(headings,-1), [1,1,NUM_SIZE_CLUSTER]) # (B,NH,NS)
    headings = headings.unsqueeze(-1).repeat(1,1,NUM_SIZE_CLUSTER)
#     print(headings.size())
#     centers = tf.tile(tf.expand_dims(tf.expand_dims(center,1),1), [1,NUM_HEADING_BIN, NUM_SIZE_CLUSTER,1]) # (B,NH,NS,3)
    centers = center.unsqueeze(1).unsqueeze(1).repeat(1,NUM_HEADING_BIN, NUM_SIZE_CLUSTER,1)
#     print(centers.size())
    N = batch_size*NUM_HEADING_BIN*NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(torch.reshape(centers, [N,3]), torch.reshape(headings, [N]), torch.reshape(sizes, [N,3]))
#     print(corners_3d.size())
    return torch.reshape(corners_3d, [batch_size, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3])

def huber_loss(error, delta):
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error,  torch.full_like(abs_error,  delta))
#     print(quadratic)
    linear = (abs_error - quadratic)
#     print(linear)
    losses = 0.5 * quadratic**2 + delta * linear
#     print(losses)
    return torch.mean(losses)


def get_loss(mask_label, center_label, \
             heading_class_label, heading_residual_label, \
             size_class_label, size_residual_label, \
             end_points, \
             writer, \
             step, \
             corner_loss_weight=10.0, \
             box_loss_weight=1.0):
    ''' Loss functions for 3D object detection.
    Input:
        mask_label: TF int32 tensor in shape (B,N)
        center_label: TF tensor in shape (B,3)
        heading_class_label: TF int32 tensor in shape (B,) 
        heading_residual_label: TF tensor in shape (B,) 
        size_class_label: TF tensor int32 in shape (B,)
        size_residual_label: TF tensor  in shape (B,)
        end_points: dict, outputs from our model
        corner_loss_weight: float scalar
        box_loss_weight: float scalar
    Output:
        total_loss: TF scalar tensor
            the total_loss is also added to the losses collection
    '''
    # 3D Segmentation loss
#     mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
#         logits=end_points['mask_logits'], labels=mask_label))
#     tf.summary.scalar('3d mask loss', mask_loss)
    loss3d = nn.CrossEntropyLoss(reduction ='mean')
    mask_loss = loss3d(end_points['mask_logits'].transpose(2,1), (mask_label) )

    # Center regression losses
#     center_dist = tf.norm(center_label - end_points['center'], axis=-1)
    center_dist = torch.norm((center_label) - end_points['center'], dim=-1)
    center_loss = huber_loss(center_dist, delta=2.0)
#     tf.summary.scalar('center loss', center_loss)
    stage1_center_dist = torch.norm((center_label) - \
        end_points['stage1_center'], dim=-1)
    stage1_center_loss = huber_loss((stage1_center_dist), delta=1.0)
#     tf.summary.scalar('stage1 center loss', stage1_center_loss)

    # Heading loss
    loss_heading = nn.CrossEntropyLoss(reduction ='mean')
#     heading_class_loss = tf.reduce_mean( \
#         tf.nn.sparse_softmax_cross_entropy_with_logits( \
#         logits=end_points['heading_scores'], labels=heading_class_label))
    heading_class_loss = loss_heading(end_points['heading_scores'], (heading_class_label) )
#     tf.summary.scalar('heading class loss', heading_class_loss)
    hcls_onehot = torch.nn.functional.one_hot(heading_class_label, NUM_HEADING_BIN) # BxNUM_HEADING_BIN
#     hcls_onehot = tf.one_hot(heading_class_label,
#         depth=NUM_HEADING_BIN,
#         on_value=1, off_value=0, axis=-1)
    heading_residual_normalized_label = \
        heading_residual_label / (np.pi/NUM_HEADING_BIN)
    temp_val = torch.sum(end_points['heading_residuals_normalized']*hcls_onehot.float(), dim=1)
    heading_residual_normalized_loss = huber_loss( temp_val - (heading_residual_normalized_label), delta=1.0)
#     tf.summary.scalar('heading residual normalized loss',
#         heading_residual_normalized_loss)

    # Size loss
    loss_size = nn.CrossEntropyLoss(reduction ='mean')
    size_class_loss = loss_size(end_points['heading_scores'], (size_class_label) )
#     size_class_loss = tf.reduce_mean( \
#         tf.nn.sparse_softmax_cross_entropy_with_logits( \
#         logits=end_points['size_scores'], labels=size_class_label))
#     tf.summary.scalar('size class loss', size_class_loss)

    scls_onehot = torch.nn.functional.one_hot(size_class_label, NUM_SIZE_CLUSTER) # BxNUM_SIZE_CLUSTER
    scls_onehot_tiled = scls_onehot.float().unsqueeze(-1).repeat(1, 1, 3)
#     tf.tile(tf.expand_dims( \
#         tf.to_float(scls_onehot), -1), [1,1,3]) # BxNUM_SIZE_CLUSTERx3
    predicted_size_residual_normalized = torch.sum( \
        end_points['size_residuals_normalized']*scls_onehot_tiled, dim=[1]) # Bx3

    mean_size_arr_expand = torch.from_numpy(g_mean_size_arr).unsqueeze(0) # 1xNUM_SIZE_CLUSTERx3
#     mean_size_label = tf.reduce_sum( \
#         scls_onehot_tiled * mean_size_arr_expand, axis=[1]) # Bx3
    mean_size_label = torch.sum(scls_onehot_tiled * mean_size_arr_expand, dim=[1]) # Bx3
    size_residual_label_normalized = size_residual_label / mean_size_label
    size_normalized_dist = torch.norm( \
        size_residual_label_normalized - predicted_size_residual_normalized, dim=-1)
    size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)
#     tf.summary.scalar('size residual normalized loss',
#         size_residual_normalized_loss)

    # Corner loss
    # We select the predicted corners corresponding to the
    # GT heading bin and size cluster.
    corners_3d = get_box3d_corners(end_points['center'],
        end_points['heading_residuals'],
        end_points['size_residuals']) # (B,NH,NS,8,3)
#     gt_mask = tf.tile(tf.expand_dims(hcls_onehot, 2), [1,1,NUM_SIZE_CLUSTER]) * \
#         tf.tile(tf.expand_dims(scls_onehot,1), [1,NUM_HEADING_BIN,1]) # (B,NH,NS)
    gt_mask = hcls_onehot.unsqueeze(2).repeat(1,1,NUM_SIZE_CLUSTER)* scls_onehot.unsqueeze(1).repeat(1,NUM_HEADING_BIN,1)

#     corners_3d_pred = tf.reduce_sum( \
#         tf.to_float(tf.expand_dims(tf.expand_dims(gt_mask,-1),-1)) * corners_3d,
#         axis=[1,2]) # (B,8,3)
    corners_3d_pred = torch.sum(gt_mask.unsqueeze(-1).unsqueeze(-1).float()*corners_3d, axis=[1,2])

    heading_bin_centers = torch.from_numpy(np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN)) # (NH,)
#     heading_label = tf.expand_dims(heading_residual_label,1) + \
#         tf.expand_dims(heading_bin_centers, 0) # (B,NH)
    heading_label = (heading_residual_label.unsqueeze(1) + heading_bin_centers.unsqueeze(0))

    heading_label = torch.sum((hcls_onehot.float())*heading_label, 1)
    mean_sizes = torch.from_numpy(g_mean_size_arr).unsqueeze(0) # (1,NS,3)

    size_label = mean_sizes + size_residual_label.unsqueeze(1)   # (1,NS,3) + (B,1,3) = (B,NS,3)
#     print(mean_sizes.size(), size_residual_label.size(), size_label.size())
    size_label = torch.sum(scls_onehot.float().unsqueeze(-1)*size_label, axis=[1]) # (B,3)
    corners_3d_gt = get_box3d_corners_helper( \
        center_label, heading_label, size_label) # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper( \
        center_label, heading_label+np.pi, size_label) # (B,8,3)

    corners_dist = torch.minimum(torch.norm(corners_3d_pred - corners_3d_gt, dim=-1),
        torch.norm(corners_3d_pred - corners_3d_gt_flip, dim=-1))
    corners_loss = huber_loss(corners_dist, delta=1.0)
#     tf.summary.scalar('corners loss', corners_loss)

    # Weighted sum of all losses
    total_loss = mask_loss + box_loss_weight * (center_loss + \
        heading_class_loss + size_class_loss + \
        heading_residual_normalized_loss*20 + \
        size_residual_normalized_loss*20 + \
        stage1_center_loss + \
        corner_loss_weight*corners_loss)
#     tf.add_to_collection('losses', total_loss)

    return total_loss



if __name__=="__main__":
    sim_data_point_cloud = Variable(torch.rand(32,2500,4))
    one_hot = Variable(torch.rand(32,3))
    print("Inputs are point cloud BxNx4 and one-hot vector Bx3")
    trans = CenterRegressionTNet()
    outstn = trans(sim_data_point_cloud, one_hot)
    print('Center Regression Net', outstn.size())


    seg = PointNetSegmentation(k = 2)
    out = seg(sim_data_point_cloud, one_hot)
    print('Segmentation', out.size())


    frustrum = FrustrumPointNent_v1()
    end_points = frustrum(sim_data_point_cloud, one_hot)
    print("All outputs:")
    for key in end_points:
        print("Size of %s :" %key, end_points[key].size() , type(end_points[key]))