import torch 
import torch.nn as nn 

torch.manual_seed(42)

DEVICE=  'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False, device = DEVICE)
        self.ln = nn.LayerNorm(10, device = DEVICE)
        self.fc2 = nn.Linear(10, out_features, bias=False, device = DEVICE)
        self.relu = nn.ReLU()

    def forward(self, x):
        print('x input dtype is ', x.dtype)
        x = self.fc1(x) # the input of fp32 is downscaled and then multiplied with weight of fp16 , to get an output value of fp16 ! 
        print('x after fc1 is ', x.dtype)
        x = self.relu(x)
        print('x after relu is ', x.dtype)
        x = self.ln(x)
        print('x after layernorm is ', x.dtype) # this comes as fp32 ( because or reduction operations aka mean and as the matrix size for reduction operation increases the value also increases)
        x = self.fc2(x)
        print('x final output is ', x.dtype)
        return x


input = torch.randn(size = (10,10), device= DEVICE)
output = 2 * input
# import random

# train this model 
model = ToyModel(in_features = 10, out_features = 10)

optimizer = torch.optim.AdamW(model.parameters())
loss_fn = torch.nn.functional.mse_loss

for epoch in range(1):
    with torch.autocast(device_type = DEVICE, dtype=torch.bfloat16):
        optimizer.zero_grad()
        out = model(input)
        print('out dtype is : ', out.dtype)
        loss = loss_fn(out, output)
        print('Loss value is ', loss, loss.dtype)

        # cast loss to fp16
        loss = loss.to(torch.float16)  
        print('Updated loss value is ', loss, loss.dtype)

        loss.backward()
        optimizer.step()

for param in model.parameters():
    if param.requires_grad:
        print('Param grad is : ', param.grad.dtype)
        break
