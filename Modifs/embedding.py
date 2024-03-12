# Copyright 2022 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        #
        self.compressed = False
        self.channels_in, self.channels_out = channels_in, channels_out

        # Normalize inputs
        self.norm = nn.BatchNorm1d(channels_in)

        self.embedding = nn.Conv1d(channels_in, channels_out, 1)

        # Point Embedding
        self.mlp = nn.Sequential(
            nn.Conv1d(channels_out, channels_out, 1),
            nn.BatchNorm1d(channels_out),
            nn.ReLU(),
            nn.Conv1d(channels_out, channels_out, 1),
            nn.BatchNorm1d(channels_out),
            nn.ReLU()
        )


        # Neighborhood embedding
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_out, 1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, 1, bias=False),
        )

        # Merge point and neighborhood embeddings
        self.final = nn.Conv1d(2 * channels_out, channels_out, 1, bias=True, padding=0)

    def transform(self, x, batch_size, nb_clusters, nmax):
        x = x.view(batch_size, nb_clusters, self.channels_out, nmax)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, self.channels_out, nb_clusters * nmax)

        labels = labels.view(batch_size * nb_clusters, 1, nmax)
        labels = labels.view(batch_size, nb_clusters, 1, nmax) 
        labels = labels.permute(0, 2, 1, 3).contiguous()  
        labels = labels.view(batch_size, 1, nb_clusters * nmax)  
        labels = labels.view(batch_size, nb_clusters * nmax, 1) # Faut s'assurer que ça ne modifie pas l'ordre mais en théorie non 

        return x



    def forward(self, x):
        """x: B x Cluster_max x Nmax x C_in . Output: B x C_out x N
        labels: B x Cluster_max x Nmax x 1 . Output: B x N x 1"""

        batch_size, nb_clusters, nmax, _ = x.shape

        # Modification organisation B x C x N x C_in -> (B x C) x C_in x N
        x = x.view(batch_size * nb_clusters, self.channels_in, nmax)
        # Normalize input x : (B x C) x C_in x N
        x = self.norm(x)

        # Point embedding : (B x C) x C_in x N -> (B x C) x C_out x N
        point_emb = self.embedding(x)

        point_tokens = self.mlp(point_emb)

        point_tokens = point_tokens + point_emb

        # Transform  (B x C) x C_out x nmax -> B x C_out x (C x nmax)
        point_tokens = self.transform(point_tokens, batch_size, nb_clusters, nmax)

        # Random downsampling to N_downsampled (with taking into account padding)
        # to implement further (if needed)
        # Si c'est implémenté, il faudra modifier les labels en conséquence, pour l'ordre des labels peut être modifié dans Collate
    
        return point_tokens
