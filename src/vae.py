import torch
import torch.nn as nn
class VAE(nn.Module):
    def __init__(self , input_dim=784, hidden_dim = 400 , latent_dim=20):
        super(VAE,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU()
        )
        self.fc_mu =nn.Linear(hidden_dim,latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim , latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,input_dim),
            nn.Sigmoid()
        )
    def encode(self, x):
        h=self.encoder(x)
        mu=self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu,logvar
    def reparameterize(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu+eps*std
    def decode(self,z):
        return self.decoder(z)
    def forward(self,x):
        mu,logvar = self.encode(x)
        z=self.reparameterize(mu,logvar)
        recon_x=self.decode(z)
        return recon_x,mu,logvar
    