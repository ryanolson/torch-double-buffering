require 'cutorch'
require 'math'
require 'os'

local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local printf = function(s,...)
  return io.write(s:format(...))
end

-- qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890
local charset = {}
for i = 48,  57 do table.insert(charset, string.char(i)) end
for i = 65,  90 do table.insert(charset, string.char(i)) end
for i = 97, 122 do table.insert(charset, string.char(i)) end

local function strrand(length)
  if length > 0 then
    return strrand(length - 1) .. charset[math.random(1, #charset)]
  else
    return ""
  end
 end

-- set up user streams

cutorch.reserveStreams(2, true)
print("number of streams: " .. cutorch.getNumStreams())

-- set up threads

local workers = Threads(4,
    function()
        require 'torch'
        require 'io'
        require 'string'
        require 'math'
    end,
    function(idx)
        tid = idx
        local seed = idx
        torch.manualSeed(seed)
        math.randomseed(seed)
        print(string.format('Starting worker with id: %d; seed: %d', tid, seed))
    end
)

-- double buffered copy and compute

local workingStream = 1
local state = {}
state[1] = nil
state[2] = nil
print("main thread currentStream: " .. cutorch.getStream())


function Compute(images, labels, streamID)
  -- this should be the training function
  -- do not use cutorch.synchronize within this function or it will cause the
  -- other data loading streams to unnecesarily sync
  -- instead, use cutorch.streamWaitFor on the cutorch.getStream
  cutorch.setStream(streamID)
  printf("Computing: images=%s labels=%s on stream=%d\n", images, labels, cutorch.getStream())
end

function CopyAndCompute(images, labels)
  -- this is the ending callback executed on the main thread
 
  -- switch streams to copy stream
  local copyStream, computeStream
  computeStream = workingStream
  workingStream = (workingStream % 2) + 1
  copyStream = workingStream

  -- setup async copy of the current finished minibatch of labels and images to
  -- a gpu tensor using :copyAsync
  cutorch.setStream(copyStream)
  printf("AsyncCopy: images=%s labels=%s on stream=%d\n", images, labels, cutorch.getStream())
 
  -- compute minibatch that was previously inflight
  if state[computeStream] then
    minibatch = state[computeStream]
    Compute(minibatch.images, minibatch.labels, computeStream)
    printf("Sync on stream=%d\n", cutorch.getStream())
  end

  -- save state
  state[copyStream] = { images=images, labels=labels }

end


for i = 1,10 do
  workers:addjob(
    function()
      -- this is the minibatch creating worker function
      -- on completion, (images, labels) are passed to CopyAndCompute
      -- which is serialized on the main thread
      local images = strrand(10)
      local labels = strrand(4)
      printf("[tid %d]: images=%s label=%s\n", __threadid, images, labels)
      return images, labels
    end,
    CopyAndCompute
  )
end
workers:synchronize()
finalComputeStream = (workingStream % 2) + 1
minibatch = state[finalComputeStream]
Compute(minibatch.images, minibatch.labels, finalComputeStream)
print("Finished")
