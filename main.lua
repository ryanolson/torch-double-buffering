require 'cutorch'

print("N: " .. arg[1])
N = tonumber(arg[1])
a_host = torch.FloatTensor(N, N)
c_host = torch.FloatTensor(N, N):zero()
b_host = { cutorch.createCudaHostTensor(N, N), cutorch.createCudaHostTensor(N, N) }

a = a_host:cuda()
c = c_host:cuda()
b = { b_host[1]:cuda(), b_host[2]:cuda() }

p = 2

totalTimer = torch.Timer()
for i = 1, 10 do
  copyTimer = torch.Timer()
  b[p]:copy(b_host[p])
--print("copyTimer: " .. copyTimer:time().real)

  p = (i % 2) + 1

  mmTimer = torch.Timer()
  torch.mm(c, a, b[p])
  cutorch.synchronize()
--print("mmTimer: " .. mmTimer:time().real)
end
print("totalTimer: " .. totalTimer:time().real)
  
