# To install this dataset you will need to have a google cloud account and a project with billing enabled.
# You will also need to have the gcloud SDK installed and configured with your account.
# You will also need to have the waymo open dataset packages installed :
# ! python3 -m pip install gcsfs waymo-open-dataset-tf-2-11-0==1.6.1

# You need to have the folder lidar, lidar_segmentation, lidar_camera_projection, lidar_calibration, stats, vehicle_pose, lidar_pose, with the right data in the current working directory.
# You need to have a folder named outputs in the current working directory.

from waymo_open_dataset.v2.perception.utils import lidar_utils
import numpy as np
import os
from typing import Optional
import warnings
# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)


import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2

from tqdm import tqdm



def read(tag: str, context_name: str) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
  return dd.read_parquet(paths)

laser_index = 1

def map_labels_to_point_cloud(points_tensor: tf.Tensor, labels_range_image: tf.Tensor, range_image_mask: tf.Tensor) -> tf.Tensor:
    """
    Maps segmentation labels to each point in the point cloud.

    Args:
        points_tensor: A [N, D] tensor of 3D LiDAR points, where N is the number of points and D is the point dimensionality.
        labels_range_image: A tensor of segmentation labels corresponding to the range image.
        range_image_mask: A boolean tensor indicating valid points in the range image.

    Returns:
        A [N,] tensor of segmentation labels corresponding to each point in the point cloud.
    """
    # Utiliser le masque de l'image de portée pour sélectionner les labels correspondants aux points valides
    valid_labels = tf.gather_nd(labels_range_image, tf.where(range_image_mask))

    return valid_labels

def processing(lidar_df, lidar_pose_df, lidar_calibration_df, lidar_segmentation_df, vehicle_pose_df, folder, laser_index = 1):
  lidar_df_filtered = lidar_df[lidar_df['key.laser_name'] == laser_index].compute()
  lidar_pose_df_filtered = lidar_pose_df[lidar_pose_df['key.laser_name'] == laser_index].compute()


  # faire une boucle sur l'ensemble des lignes sélectionnées (lidar_segmentation sans doute, du coup pas besoin de trier avec le laser 1)
  vehicle_pose_df_pd = vehicle_pose_df.compute()
  lidar_segmentation_df = lidar_segmentation_df.compute()
  lidar_calibration = lidar_calibration_df[lidar_calibration_df['key.laser_name'] == laser_index].compute().iloc[0].to_dict()
  lidar_calibration = v2.LiDARCalibrationComponent.from_dict(lidar_calibration)
  for index in lidar_segmentation_df.index :
    vehicle_pose = v2.VehiclePoseComponent.from_dict(vehicle_pose_df_pd.loc[index].to_dict())
    lidar = v2.LiDARComponent.from_dict(lidar_df_filtered.loc[index].to_dict())
    lidar_pose = v2.LiDARPoseComponent.from_dict(lidar_pose_df_filtered.loc[index].to_dict())
    lidar_segmentation = v2.LiDARSegmentationLabelComponent.from_dict(lidar_segmentation_df.loc[index].to_dict())
    points1 = lidar_utils.convert_range_image_to_point_cloud(lidar.range_image_return1,
                                                            lidar_calibration,
                                                            lidar_pose.range_image_return1,
                                                            vehicle_pose,
                                                            True)
    points2 = lidar_utils.convert_range_image_to_point_cloud(lidar.range_image_return2,
                                                            lidar_calibration,
                                                            lidar_pose.range_image_return1,
                                                            vehicle_pose,
                                                            True)
    shapes1 = lidar_segmentation.range_image_return1.shape

    point_cloud = np.concatenate((points1.numpy(), points2.numpy()), axis=0)
    point_cloud = point_cloud[:,[3,4,5,1]] # On récupère (x,y,z, intensity) la range étant calculé dans le dataloader

    labels_range_image = tf.convert_to_tensor(lidar_segmentation.range_image_return1.values.reshape(shapes1[0], shapes1[1], 2)[:,:,1])

    range_image_mask = tf.convert_to_tensor(lidar.range_image_return1.values.reshape(shapes1[0], shapes1[1], 4))[..., 0] > 0
    seg1 = map_labels_to_point_cloud(points1, labels_range_image, range_image_mask)
    shapes2 = lidar_segmentation.range_image_return2.shape
    labels_range_image = tf.convert_to_tensor(lidar_segmentation.range_image_return2.values.reshape(shapes2[0], shapes2[1], 2)[:,:,1])
    range_image_mask = tf.convert_to_tensor(lidar.range_image_return2.values.reshape(shapes2[0], shapes2[1], 4))[..., 0] > 0
    seg2 = map_labels_to_point_cloud(points2, labels_range_image, range_image_mask)
    segmentation = np.concatenate((seg1.numpy(), seg2.numpy()), axis=0, dtype=np.int16)
    segmentation.tofile(os.path.join(folder,f'{index}.label'))
    point_cloud.astype(np.float32).tofile(os.path.join(folder,f'{index}.bin'))



outputs = 'outputs'
list_context = [f.split('.')[0] for f in os.listdir("./lidar_camera_projection") if f.endswith(".parquet")]
deja = [f.split(';')[0] for f in os.listdir(outputs)]
for context in tqdm(list_context):
    if context not in deja:
        lidar_calibration_df = read('lidar_calibration', context)
        lidar_df = read('lidar', context)
        lidar_pose_df = read('lidar_pose', context)
        lidar_camera_df = read('lidar_camera_projection', context)
        lidar_segmentation_df = read('lidar_segmentation', context)
        vehicle_pose_df = read('vehicle_pose', context)
        processing(lidar_df, lidar_pose_df, lidar_calibration_df, lidar_segmentation_df, vehicle_pose_df, outputs, laser_index = 1)

