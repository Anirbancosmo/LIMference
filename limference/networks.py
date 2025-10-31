"""
Neural network architectures for LIMference package
Includes U-Net, ResNet, and embedding networks for field-level inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union
import numpy as np
from sbi.neural_nets.embedding_nets import CNNEmbedding


class UNet(nn.Module):
    """
    U-Net architecture for field-level analysis and denoising
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        init_features: Initial number of features (doubled at each level)
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 32,
        dropout_rate: float = 0.0
    ):
        super(UNet, self).__init__()
        
        features = init_features
        self.dropout_rate = dropout_rate
        
        # Encoder path
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")
        
        # Decoder path
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block(features * 16, features * 8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features * 8, features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        
        # Final convolution
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)
        
        # Dropout layers if specified
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
    
    def _block(self, in_channels: int, features: int, name: str) -> nn.Sequential:
        """Create a basic convolutional block"""
        layers = [
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        ]
        
        if self.dropout_rate > 0 and "enc" in name:  # Add dropout in encoder only
            layers.append(nn.Dropout2d(self.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net"""
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final output
        return torch.sigmoid(self.conv(dec1))


class ResNetBlock(nn.Module):
    """Basic ResNet block with skip connection"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet architecture for deterministic predictions
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes/parameters
        init_features: Initial number of features
        blocks: List of number of blocks in each layer
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        init_features: int = 64,
        blocks: List[int] = [2, 2, 2, 2]
    ):
        super(ResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, init_features, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(init_features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(init_features, init_features, blocks[0])
        self.layer2 = self._make_layer(init_features, init_features * 2, blocks[1], stride=2)
        self.layer3 = self._make_layer(init_features * 2, init_features * 4, blocks[2], stride=2)
        self.layer4 = self._make_layer(init_features * 4, init_features * 8, blocks[3], stride=2)
        
        # Global average pooling and fully connected
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(init_features * 8, num_classes)
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a ResNet layer with multiple blocks"""
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class ResNetProbabilistic(nn.Module):
    """
    Probabilistic ResNet for parameter estimation with uncertainty
    
    Args:
        in_channels: Number of input channels
        num_params: Number of parameters to predict
        init_features: Initial number of features
        blocks: List of number of blocks in each layer
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_params: int = 2,
        init_features: int = 64,
        blocks: List[int] = [2, 2, 2, 2]
    ):
        super(ResNetProbabilistic, self).__init__()
        
        # Use base ResNet as feature extractor
        self.features = ResNet(in_channels, num_params * 2, init_features, blocks)
        
        # Separate heads for mean and variance
        self.fc_mean = nn.Linear(init_features * 8, num_params)
        self.fc_log_var = nn.Linear(init_features * 8, num_params)
        
        # Override the final FC layer of ResNet
        self.features.fc = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning mean and variance
        
        Returns:
            Tuple of (mean, variance) tensors
        """
        # Extract features
        features = self.features(x)
        
        # Predict mean and log variance
        mean = self.fc_mean(features)
        log_var = self.fc_log_var(features)
        
        # Convert log variance to variance (ensure positive)
        var = F.softplus(log_var) + 1e-6
        
        return mean, var
    
    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Sample from the predicted distribution
        
        Args:
            x: Input tensor
            n_samples: Number of samples to draw
            
        Returns:
            Samples from the predicted distribution
        """
        mean, var = self.forward(x)
        std = torch.sqrt(var)
        
        # Sample from the distribution
        batch_size = x.shape[0]
        samples = []
        
        for _ in range(n_samples):
            eps = torch.randn_like(mean)
            sample = mean + std * eps
            samples.append(sample)
        
        if n_samples == 1:
            return samples[0]
        else:
            return torch.stack(samples, dim=1)


class AttentionBlock(nn.Module):
    """Self-attention block for field-level analysis"""
    
    def __init__(self, in_channels: int):
        super(AttentionBlock, self).__init__()
        
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention"""
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, channels, height * width)
        
        # Calculate attention
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Apply residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


class CNNWithAttention(nn.Module):
    """
    CNN with attention mechanism for field-level inference
    
    Args:
        input_shape: Shape of input images (height, width)
        embedding_dim: Dimension of output embedding
        use_attention: Whether to use attention blocks
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (256, 256),
        embedding_dim: int = 128,
        use_attention: bool = True
    ):
        super(CNNWithAttention, self).__init__()
        
        self.use_attention = use_attention
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Attention blocks
        if use_attention:
            self.attention1 = AttentionBlock(64)
            self.attention2 = AttentionBlock(128)
        
        # Calculate size after convolutions
        test_input = torch.zeros(1, 1, *input_shape)
        test_output = self._forward_conv(test_input)
        conv_output_size = test_output.numel()
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, embedding_dim)
        self.dropout = nn.Dropout(0.5)
    
    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers only"""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        if self.use_attention:
            x = self.attention1(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        if self.use_attention:
            x = self.attention2(x)
        
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self._forward_conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class FieldLevelEmbedding(nn.Module):
    """
    Custom embedding network for field-level SBI
    
    Args:
        input_shape: Shape of input fields
        embedding_dim: Dimension of output embedding
        architecture: Type of architecture ('unet', 'resnet', 'cnn', 'cnn_attention')
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int] = (256, 256),
        embedding_dim: int = 128,
        architecture: str = 'cnn'
    ):
        super(FieldLevelEmbedding, self).__init__()
        
        self.architecture = architecture
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        
        if architecture == 'unet':
            self.network = UNet(1, embedding_dim // 32, 32)
            # Add pooling and FC to get fixed-size embedding
            self.pool = nn.AdaptiveAvgPool2d((8, 8))
            self.fc = nn.Linear(8 * 8 * (embedding_dim // 32), embedding_dim)
            
        elif architecture == 'resnet':
            self.network = ResNet(1, embedding_dim, 64)
            
        elif architecture == 'cnn_attention':
            self.network = CNNWithAttention(input_shape, embedding_dim, use_attention=True)
            
        else:  # Default CNN
            self.network = CNNEmbedding(input_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to create embedding"""
        # Add batch and channel dimensions if needed
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        
        if self.architecture == 'unet':
            x = self.network(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = self.network(x)
        
        return x


# Factory function for creating networks
def create_network(
    network_type: str,
    config: Optional[dict] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create neural networks
    
    Args:
        network_type: Type of network ('unet', 'resnet', 'resnet_prob', 'cnn_attention', 'field_embedding')
        config: Configuration dictionary
        **kwargs: Additional arguments for the network
        
    Returns:
        Neural network module
    """
    if config:
        kwargs.update(config)
    
    if network_type == 'unet':
        return UNet(**kwargs)
    elif network_type == 'resnet':
        return ResNet(**kwargs)
    elif network_type == 'resnet_prob':
        return ResNetProbabilistic(**kwargs)
    elif network_type == 'cnn_attention':
        return CNNWithAttention(**kwargs)
    elif network_type == 'field_embedding':
        return FieldLevelEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
