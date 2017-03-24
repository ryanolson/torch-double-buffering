require 'cutorch'

N = 11150

a_host = torch.FloatTensor(N, N)
c_host = torch.FloatTensor(N, N):zero()
b_host = { torch.FloatTensor(N, N):zero(), torch.FloatTensor(N, N):zero() }

a = a_host:cuda()
c = c_host:cuda()
b = { b_host[1]:cuda(), b_host[2]:cuda() }

p = 2

print(b_host[1]:size())
print(b[1]:size())

totalTimer = torch.Timer()
for i = 1, 10 do
  p = (i % 2) + 1
  copyTimer = torch.Timer()
  b[p]:copy(b_host[p])
  cutorch.synchronize()
  print("copyTimer: " .. copyTimer:time().real)

  mmTimer = torch.Timer()
  torch.mm(c, a, b[p])
  cutorch.synchronize()
  print("mmTimer: " .. mmTimer:time().real)
end
print("totalTimer: " .. totalTimer:time().real)
  
