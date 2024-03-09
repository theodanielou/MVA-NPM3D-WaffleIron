import os
import numpy as np
from .pc_dataset import PCDataset

class WaymoSemSeg(PCDataset):
    CLASS_NAME =["Car", "Truck", "Bus", "Motorcyclist", "Bicyclist", "Pedestrian", "Sign", "Traffic Light", "Pole", "Construction Cone", "Bicycle", "Motorcycle", "Building", "Vegetation", "Tree Trunk", "Curb", "Road", "Lane Marker", "Walkable", "Sidewalk", "Other Ground", "Other vehicle"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        current_folder = os.path.dirname(os.path.realpath(__file__))
        # List all keyframes
        self.list_frames = np.load(
            os.path.join(current_folder, "list_files_waymo.npz") # à créer
        )[self.phase]
        if self.phase == "train":
            assert len(self) == 3736
        elif self.phase == "val":
            assert len(self) == 934
        # elif self.phase == "test":
        #     assert len(self) == 6008
        elif self.phase == "test":
            assert len(self) == 519
        else:
            raise ValueError(f"Unknown phase {self.phase}.")

        assert not self.instance_cutmix, "Instance CutMix not implemented on Waymo"

    def __len__(self):
        return len(self.list_frames)

    def load_pc(self, index):
        # Load point cloud
        pc = np.fromfile(
            os.path.join(self.rootdir, self.list_frames[index][0]),
            dtype=np.float32,
        )
        pc = pc.reshape((-1, 4))

        # Load segmentation labels
        labels = np.fromfile(
            os.path.join(self.rootdir, self.list_frames[index][1]),
            dtype=np.int16,
        )

        labels = labels - 1
        labels[labels == -1] = 255

        return pc, labels, self.list_frames[index][2]