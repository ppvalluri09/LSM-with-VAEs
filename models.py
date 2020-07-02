import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            # mx150x150x1
            nn.Conv2d(1, 16, 4, 2, 0),
            nn.ReLU(),
            # mx74x74x16
            nn.Conv2d(16, 16, 4, 2, 0),
            nn.ReLU(),
            # mx36x36x16
            nn.Conv2d(16, 32, 4, 2, 0),
            # mx17x17x32
            nn.Conv2d(32, 8, 3, 2, 0)
            # mx8x8x8
        )
        
        self.mean = nn.Linear(8*8*8, 40)
        self.std = nn.Linear(8*8*8, 40)
        
        self.convertor = nn.Linear(40, 8*8*8)
        
        self.decoder = nn.Sequential(
            # mx8x8x8
            nn.ConvTranspose2d(8, 32, 3, 2, 0),
            nn.LeakyReLU(0.2),
            # mx17x17x16
            nn.ConvTranspose2d(32, 16, 4, 2, 0),
            nn.LeakyReLU(0.1),
            # mx36x36x16
            nn.ConvTranspose2d(16, 16, 4, 2, 0),
            nn.LeakyReLU(0.2),
            # mx74x74x16
            nn.ConvTranspose2d(16, 1, 4, 2, 0)
            # mx150x150x1
        )
        
    def encode(self, x):
        encoded = F.relu(self.encoder(x))#.view(-1, 8*8*8)
        encoded = encoded.view(-1, 8*8*8)
        return self.mean(encoded), self.std(encoded)
    
    def reparametrize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        z = torch.randn_like(std)
        return mean + z*std
    
    def decode(self, x):
        decoded = F.relu(self.convertor(x)).view(-1, 8, 8, 8)
        return F.sigmoid(self.decoder(decoded))
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparametrize(mean, logvar)
        decoded = self.decode(z)
        return decoded, mean, logvar
