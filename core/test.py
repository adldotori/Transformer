import torch

loss = torch.nn.CrossEntropyLoss()

input = torch.randn(3, 4, 5)
target = torch.randint(5, (3,4))

# input = input.view(-1, input.shape[-1])
target = torch.transpose(target, 0, 1)
print(target.shape)
target = target.contiguous().view(-1)
print(input.shape, target.shape)
# target = torch.empty(3, dtype=torch.long).random_(5)

output = loss(input, target)
print(input)
print(target)
print(output)