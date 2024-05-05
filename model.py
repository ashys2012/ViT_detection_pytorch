import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class mlp(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(mlp, self).__init__()
        self.layers = nn.ModuleList()  # A list to hold all layers
        
        # Add each layer defined by hidden_units
        for units in hidden_units:
            self.layers.append(nn.Linear(units[0], units[1]))  # nn.Linear instead of Dense
            self.layers.append(nn.Dropout(dropout_rate))  # Dropout layer

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = F.gelu(layer(x))  # Apply GELU activation function after Linear
            else:
                x = layer(x)  # Apply dropout
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class Patches(nn.Module):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def forward(self, images):
        # images expected shape: [B, C, H, W]
        batch_size, channels, height, width = images.size()

        # Calculate number of patches along height and width
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        # Extract patches
        #    Extract Patches Along Height (unfold(2, self.patch_size, self.patch_size)):
    #     This call slides a window of size self.patch_size over the height of the image (
    #dimension 2).  It extracts slices of height self.patch_size with a stride of 
    #self.patch_size, meaning there's no overlap between consecutive slices.
    #  After this operation, each slice is a partial patch containing the entire 
    #width but only a portion of the height.

    # Extract Patches Along Width (unfold(3, self.patch_size, self.patch_size)):
    # This follows the first unfold and applies to the width (dimension 3).
    # It takes the output from the previous unfold, which has strips of the image, 
    # and further slices them into full patches that are self.patch_size by self.patch_size.


        patches = images.unfold(2, self.patch_size, self.patch_size) \
                         .unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(batch_size, num_patches_h * num_patches_w, channels * self.patch_size * self.patch_size)

        return patches   # shape is [B, N, C*patch_size*patch_size] or channels * patch_size * patch_size

    def get_config(self):
        # Optional method to help with more detailed serialization of the model
        return {'patch_size': self.patch_size}


import torch
import torch.nn as nn

class PatchEncoder(nn.Module):
    def __init__(self, num_patches, patch_size, channels, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.channels = channels
        self.projection_dim = projection_dim
        self.patch_length = channels * patch_size * patch_size  # dynamically computed patch length
        
        # Set up the projection layer with the dynamic input dimension
        self.projection = nn.Linear(self.patch_length, self.projection_dim)
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, projection_dim))

    def forward(self, x):
        # x: [batch_size, num_patches, patch_length]
        # Apply linear projection to each patch
        x = self.projection(x)
        # Add positional embeddings to the projected patches
        x += self.position_embeddings
        return x

    def extra_repr(self):
        # Provides additional information when printing the model
        return f'num_patches={self.num_patches}, projection_dim={self.projection_dim}, patch_length={self.patch_length}'

# # Example usage
# channels = 3  # Number of channels in the image, typically 3 for RGB
# patch_size = 32  # Define the size of each patch
# num_patches = (224 // patch_size) * (224 // patch_size)  # Calculate the number of patches from image dimensions
# projection_dim = 768  # A typical dimension to project onto for transformers

# # Assuming the Patches module is defined as provided
# patches_module = Patches(patch_size=patch_size)
# patch_encoder = PatchEncoder(num_patches=num_patches, patch_size=patch_size, channels=channels, projection_dim=projection_dim)

# # Simulate input
# images = torch.randn(10, channels, 224, 224)  # Batch of 10 images, 3 channels, 224x224 size

# # Forward through the modules
# patches = patches_module(images)  # Extract patches
# encoded_patches = patch_encoder(patches)  # Encode patches

# print(encoded_patches.shape)  # Should print something like [10, 49, 768]

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_shape, patch_size, num_patches, projection_dim, num_heads, mlp_dim, transformer_layers, mlp_head_units):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        
        self.patches = Patches(patch_size)
        self.encoder = PatchEncoder(num_patches, patch_size, 3, projection_dim)  # Assuming RGB images
        
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(projection_dim, num_heads, mlp_dim, dropout=0.1) for _ in range(transformer_layers)]
        )
        
        self.head = nn.Sequential(
            nn.LayerNorm(projection_dim),
            Rearrange('b n d -> b (n d)'),
            nn.Linear(num_patches * projection_dim, mlp_head_units[0]),
            nn.GELU(),
            nn.Linear(mlp_head_units[0], mlp_head_units[1]),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_head_units[1], 4),  # Output 4 coordinates for bounding box
            nn.Softplus()
        )

    def forward(self, x):
        x = self.patches(x)
        x = self.encoder(x)
        x = self.transformer_blocks(x)
        print("x is", x)
        x = self.head(x)
        #x = nn.Softplus(x)  
        #print("x is", x)
        return x
