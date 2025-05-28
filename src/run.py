import torch
import torch.optim as optim
from vae import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

def vae_loss(recon_x,x,mu,logvar):
    BCE=nn.functional.binary_cross_entropy(recon_x,x,reduction='sum')
    KLD=-0.5*torch.sum(1+logvar -mu.pow(2) - logvar.exp() )
    return BCE + KLD

transform=transforms.Compose([
        transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data',train=True,transform=transform,download=True)
train_loader = DataLoader(train_dataset,batch_size=64 , shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(),lr=1e-3)

epochs=10
if not os.path.exists('./results'):
    os.makedirs('./results')
if not os.path.exists('./models'):
    os.makedirs('./models')

vae.train()
for epoch in range(1,epochs+1):
    train_loss=0
    for batch_idx , (data, _) in enumerate(train_loader):
        data=data.view(-1,784).to(device)
        optimizer.zero_grad()
        recon_batch,mu,logvar = vae(data)
        loss=vae_loss(recon_batch , data , mu, logvar)
        loss.backward()
        train_loss+=loss.item()
        optimizer.step()

    average_loss  = train_loss / len(train_loader.dataset)
    print(f'Epoch {epoch} , Average Loss:{average_loss:.4f}')

    with torch.no_grad():
        z=torch.randn(64,20).to(device)
        sample = vae.decode(z).cpu()
        save_image(sample.view(64,1,28,28), f'./results/sample_epoch_{epoch}.png')
    
    torch.save(vae,f"./models/vae_epoch_{epoch}.pth")

vae.eval()
with torch.no_grad():
    z = torch.randn(16, 20).to(device)
    generated = vae.decode(z).cpu()
    save_image(generated.view(16, 1, 28, 28), 'generated_digits.png')
