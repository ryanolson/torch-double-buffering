require 'cutorch'

print("N: " .. arg[1])
N = tonumber(arg[1])
a_host = torch.FloatTensor(N, N)
c_host = torch.FloatTensor(N, N):zero()
b_host = { cutorch.createCudaHostTensor(N, N), cutorch.createCudaHostTensor(N, N) }

a = a_host:cuda()
c = c_host:cuda()
b = { b_host[1]:cuda(), b_host[2]:cuda() }

p = 1

cutorch.reserveStreams(2, true)
print("number of streams: " .. cutorch.getNumStreams())

totalTimer = torch.Timer()
for i = 1, 10 do
  cutorch.setStream(p)
  b[p]:copyAsync(b_host[p])

  p = (i % 2) + 1

  cutorch.setStream(p)
  torch.mm(c, a, b[p])
  cutorch.synchronizeAll()
end
print("totalTimer: " .. totalTimer:time().real)
  
