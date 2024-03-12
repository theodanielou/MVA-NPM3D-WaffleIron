import torch
import numpy as np
import utils.transforms as tr
from torch.utils.data import Dataset
import torch.nn.functional as F # Ajout
from scipy.spatial import cKDTree as KDTree
from sklearn.cluster import KMeans, DBSCAN # Ajout


class PCDataset(Dataset):
    def __init__(
        self,
        rootdir=None,
        phase="train",
        input_feat="intensity",
        voxel_size=0.1,
        train_augmentations=None,
        dim_proj=[
            0,
        ],
        grids_shape=[(256, 256)],
        fov_xyz=(
            (
                -1.0,
                -1.0,
                -1.0,
            ),
            (1.0, 1.0, 1.0),
        ),
        num_neighbors=16, # N'est plus utile car on a un nombre fixe de clusters
        tta=False,
        instance_cutmix=False,
        nmax=50,
        cmax=300,
    ):
        super().__init__()

        # Dataset split
        self.phase = phase
        assert self.phase in ["train", "val", "trainval", "test"]

        # Root directory of dataset
        self.rootdir = rootdir

        # Input features to compute for each point
        self.input_feat = input_feat

        # Downsample input point cloud by small voxelization
        self.downsample = tr.Voxelize(
            dims=(0, 1, 2),
            voxel_size=voxel_size,
            random=(self.phase == "train" or self.phase == "trainval"),
        )

        # Field of view
        assert len(fov_xyz[0]) == len(
            fov_xyz[1]
        ), "Min and Max FOV must have the same length."
        for i, (min, max) in enumerate(zip(*fov_xyz)):
            assert (
                min < max
            ), f"Field of view: min ({min}) < max ({max}) is expected on dimension {i}."
        self.fov_xyz = np.concatenate([np.array(f)[None] for f in fov_xyz], axis=0)
        self.crop_to_fov = tr.Crop(dims=(0, 1, 2), fov=fov_xyz)

        # Grid shape for projection in 2D
        assert len(grids_shape) == len(dim_proj)
        self.dim_proj = dim_proj
        self.grids_shape = [np.array(g) for g in grids_shape]
        self.lut_axis_plane = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

        # Number of neighbors for embedding layer
        assert num_neighbors > 0
        self.num_neighbors = num_neighbors

        # Test time augmentation
        if tta:
            assert self.phase in ["test", "val"]
            self.tta = tr.Compose(
                (
                    tr.Rotation(inplace=True, dim=2),
                    tr.Rotation(inplace=True, dim=6),
                    tr.RandomApply(tr.FlipXY(inplace=True), prob=2.0 / 3.0),
                    tr.Scale(inplace=True, dims=(0, 1, 2), range=0.1),
                )
            )
        else:
            self.tta = None

        # Train time augmentations
        if train_augmentations is not None:
            assert self.phase in ["train", "trainval"]
        self.train_augmentations = train_augmentations

        # Flag for instance cutmix
        self.instance_cutmix = instance_cutmix

        # Maximum number of points per cluster
        self.nmax = nmax
        # Maximum number of clusters
        self.cmax = cmax # 400 pour 20 000, 300 pour 15 000

    def get_occupied_2d_cells(self, pc):
        """Return mapping between 3D point and corresponding 2D cell"""
        cell_ind = []
        for dim, grid in zip(self.dim_proj, self.grids_shape):
            # Get plane of which to project
            dims = self.lut_axis_plane[dim]
            # Compute grid resolution
            res = (self.fov_xyz[1, dims] - self.fov_xyz[0, dims]) / grid[None]
            # Shift and quantize point cloud
            pc_quant = ((pc[:, dims] - self.fov_xyz[0, dims]) / res).astype("int")
            # Check that the point cloud fits on the grid
            min, max = pc_quant.min(0), pc_quant.max(0)
            assert min[0] >= 0 and min[1] >= 0, print(
                "Some points are outside the FOV:", pc[:, :3].min(0), self.fov_xyz
            )
            assert max[0] < grid[0] and max[1] < grid[1], print(
                "Some points are outside the FOV:", pc[:, :3].min(0), self.fov_xyz
            )
            # Transform quantized coordinates to cell indices for projection on 2D plane
            temp = pc_quant[:, 0] * grid[1] + pc_quant[:, 1]
            cell_ind.append(temp[None])
        return np.vstack(cell_ind)

    def prepare_input_features(self, pc_orig):
        # Concatenate desired input features to coordinates
        pc = [pc_orig[:, :3]]  # Initialize with coordinates
        for type in self.input_feat:
            if type == "intensity":
                pc.append(pc_orig[:, 3:])
            elif type == "height":
                pc.append(pc_orig[:, 2:3])
            elif type == "radius":
                r_xyz = np.linalg.norm(pc_orig[:, :3], axis=1, keepdims=True)
                pc.append(r_xyz)
            elif type == "xyz":
                pc.append(pc_orig[:, :3])
            else:
                raise ValueError(f"Unknown feature: {type}")
        return np.concatenate(pc, 1)

    def load_pc(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def ajuster_clusters(self, points, indices_clusters, n_max, c_max): # Voir si on peut enlever les calculs sur clusters_temporaires pour gagner de la memoire
        clusters_temporaires = []
        indices_temporaires = []
        indices_clusters_uniques = np.unique(indices_clusters)
        
        # Diviser les clusters trop grands
        for cluster in indices_clusters_uniques:
            indices = np.where(indices_clusters == cluster)[0]
            points_cluster = points[indices]
            
            if len(points_cluster) < n_max:
                continue
            
            nombre_de_sous_clusters = len(points_cluster) // n_max
            if nombre_de_sous_clusters == 0:
                continue
            
            kmeans = KMeans(n_clusters=nombre_de_sous_clusters, n_init = 'auto', random_state=42).fit(points_cluster)
            for i in range(nombre_de_sous_clusters):
                sous_indices = indices[kmeans.labels_ == i]
                if len(sous_indices) >= n_max:
                    clusters_temporaires.append(points[sous_indices[:n_max]])
                    indices_temporaires.append(sous_indices[:n_max])

        # Calcul de la répartition des suppressions de clusters, en pratique jamais utilisé car on ne dépasse jamais c_max
        if len(clusters_temporaires) > c_max:
            surplus = len(clusters_temporaires) - c_max
            taille_clusters_originaux = {cluster: len(np.where(indices_clusters == cluster)[0]) for cluster in indices_clusters_uniques}
            total_points = sum(taille_clusters_originaux.values())
            proportions = {cluster: taille/total_points for cluster, taille in taille_clusters_originaux.items()}
            
            # Calcul des quotas de suppression par cluster original
            suppressions_par_cluster = {cluster: int(np.ceil(surplus * proportion)) for cluster, proportion in proportions.items()}
            
            # Application des suppressions en priorisant les plus grands clusters
            indices_temporaires_apres_suppression = []
            for indice_temp in indices_temporaires:
                cluster_orig = indices_clusters[indice_temp[0]]
                if suppressions_par_cluster[cluster_orig] > 0:
                    suppressions_par_cluster[cluster_orig] -= 1
                else:
                    indices_temporaires_apres_suppression.append(indice_temp)
            
            # Reconstruction des clusters après suppression
            indices_temporaires = indices_temporaires_apres_suppression

        return indices_temporaires 


    def __getitem__(self, index): # Vérifier à quel moment on un tableau numpy et à quel moment on a un tensor !
        # Load original point cloud
        pc_orig, labels_orig, filename = self.load_pc(index)

        # Prepare input feature
        pc_orig = self.prepare_input_features(pc_orig) # les 3 premiers features sont les coordonnées, qu'on utilisera pas après, le reste les features dans l'ordre (intensity, xyz, radius)
        # pc_orig : [x,y,z,intensity,xyz,radius]

        # Test time augmentation => à voir si on garde ça, à priori je l'utilise pas
        if self.tta is not None:
            pc_orig, labels_orig = self.tta(pc_orig, labels_orig)

        # Voxelization => On garde mais attention à ne pas mettre de valeurs trop grandes pour la résolution
        pc, labels = self.downsample(pc_orig, labels_orig)

        # Augment data = > à modifier pour ne pas limiter les points à Nmax
        if self.train_augmentations is not None:
            pc, labels = self.train_augmentations(pc, labels)

        # Crop to fov => ok 
        pc, labels = self.crop_to_fov(pc, labels)

        # 0.5 peut être modifié pour changer la hauteur de la séparation
        indices_ground = np.where(pc[:, 2] < 0.5)[0]
        pc_ground = pc[indices_ground, :3] # récupérer que les points pas les autres features 
        indices_high = np.where(pc[:, 2] >= 0.5)[0]
        pc_high = pc[indices_high, :3]

        # Clustering => Modifier eps, peut être le rendre modifiable dans les paramètres
        clusters_ground = DBSCAN(eps = 0.8, min_samples=20).fit_predict(pc_ground)
        clusters_high = DBSCAN(eps = 0.8, min_samples=20).fit_predict(pc_high)
        
        # Fusionner les clusters en gardant les indices initiaux de pc 
        clusters_ground += 1
        clusters_high += 1
        clusters_high += clusters_ground.max()

        clusters = np.full(shape=len(pc), fill_value=-1, dtype=int)

        clusters[indices_ground] = clusters_ground
        clusters[indices_high] = clusters_high

        # Application de la fonction
        indices_pc_in_clusters = self.ajuster_clusters(pc[:,:3], clusters, self.nmax, self.cmax) # Dimension ici [c_i, n_max, 3], [c_i, n_max], c_i<=c_max
        indices_pc_in_clusters = np.vstack(indices_pc_in_clusters)

        pc_clusters = pc[indices_pc_in_clusters, 3:] # Dimension ici [c_i, n_max, 5]

        pc_applati = pc_clusters.reshape(-1, 5) # Dimension ici [c_i * n_max, 5]
    
        labels_applati = labels[indices_pc_in_clusters.flatten()] # Dimension ici [c_i * n_max], correspond aux labels de chaque point dans pc_applati
        
        # Nearest neighbor interpolation to undo cropping & voxelisation at validation time
        if self.phase in ["train", "trainval"]:
            upsample = np.arange(pc_applati.shape[0])
        else:
            kdtree = KDTree(pc_applati[:, :3])
            _, upsample = kdtree.query(pc_orig[:, 1:4], k=1)  # Pour chaque point de pc_orig, on récupère le point l'indice du point le plus proche dans pc_aplati


        # Append padding pour avoir le bon nombre de cluster 
        c_i = pc_clusters.shape[0] # c_i
        nombre_points_padding  = (self.cmax - c_i) * self.nmax
        pc_applati_pad = np.pad(pc_applati, ((0, 0), (0, nombre_points_padding)), mode='constant', constant_values=0)
        labels_padding = np.full(nombre_points_padding, -1)
        labels_applati_pad = np.concatenate((labels_applati, labels_padding))

        pc_clusters_pad = pc_applati_pad.reshape(self.cmax, self.nmax, 5) # Dimension ici [c_max, n_max, 5]

        # Liste des index des points dans chaque cluster en prenant en compte le padding
        nouveau_index_pc_in_cluster = np.full((self.cmax, self.nmax), -1)
        # nouveau_index_pc_in_cluster[:c_i, :] = indices_pc_in_clusters
        nouveau_index_pc_in_cluster[:c_i, :] = np.arange(0, c_i * self.nmax).reshape(c_i, self.nmax)

        # Projection 2D -> 3D: index of 2D cells for each point
        cell_ind = self.get_occupied_2d_cells(pc_applati[:, 1:4])
        cell_ind = np.hstack(cell_ind, np.zeros((cell_ind.shape[0], nombre_points_padding)))

        # Occupied cells
        occupied_cells = np.ones((1, self.cmax * self.nmax))
        occupied_cells[:, c_i * self.nmax:] = 0

        # Output to return
        out = ( 
            # Point features
            pc_clusters_pad[None], # Dimension ici [1, c_max, n_max, 5]
            # Point labels of original entire point cloud
            labels_applati_pad if self.phase in ["train", "trainval"] else labels_orig, # Dimension ici [c_max * n_max] ou [n]
            # Projection 2D -> 3D: index of 2D cells for each point
            cell_ind[None], # Dimension ici [1, c_max, n_max]
            # Index to match pc_applati_pad
            nouveau_index_pc_in_cluster[None], # Dimension ici [c_max, n_max]
            # Occupied cells (for padding)
            occupied_cells, # Dimension ici [1, c_max * n_max]
            # For interpolation from voxelized & cropped point cloud to original point cloud
            upsample, # Dimension ici [n]
            # Filename of original point cloud
            filename,
        )

        return out


def transfom_index(idx, batch_size, nb_clusters, nmax):
    idx = idx.view(batch_size * nb_clusters, 1, nmax)
    idx = idx.view(batch_size, nb_clusters, 1, nmax) 
    idx = idx.permute(0, 2, 1, 3).contiguous()  
    idx = idx.view(batch_size, 1, nb_clusters * nmax)  
    idx = idx.squeeze(1)
    return idx

class Collate:
    def __init__(self, num_points=None):
        self.num_points = num_points
        assert num_points is None or num_points > 0

    def __call__(self, list_data):
        # Extract all data
        list_of_data = (list(data) for data in zip(*list_data))

        feat, label_orig, cell_ind, idx, occupied_cells, upsample, filename = list_of_data

        # Concatenate along batch dimension
        feat = torch.from_numpy(np.vstack(feat)).float()  # B x C x Nmax x 5
        cell_ind = torch.from_numpy(
            np.vstack(cell_ind)
        ).long()  # B x nb_2d_cells x (Cmax x Nmax)
        occupied_cells = torch.from_numpy(np.vstack(occupied_cells)).float()  # B x (Cmax x Nmax)
        labels_orig = torch.from_numpy(np.hstack(label_orig)).long() # B x N
        upsample = [torch.from_numpy(u) for u in upsample]

        idx = torch.from_numpy(np.vstack(idx)).long() # B x (Cmax x Nmax) 
        
        # Transform index pour récupérer la bonne forme à la sortie du modèle
        batch_size, nb_clusters, nmax, _ = feat.shape
        index_tokens = transfom_index(idx, batch_size, nb_clusters, nmax)


        # Prepare output variables
        out = {
            "feat": feat,
            "idx": index_tokens,
            "upsample": upsample,
            "labels_orig": labels_orig,
            "cell_ind": cell_ind,
            "occupied_cells": occupied_cells,
            "filename": filename,
        }

        return out
