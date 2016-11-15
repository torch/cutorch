local Threads = require 'threads'
require 'cutorch'

local function test_cudaEvent()
   cutorch.reserveStreams(2)
   cutorch.setStream(1)

   local t1 = torch.CudaTensor(10000000):zero()
   local t2 = torch.CudaTensor(1):zero()

   local t1View = t1:narrow(1, 10000000, 1)
   t1:fill(1)
   print('Memory usage after some allocations [free memory], [total memory]')
   print(cutorch.getMemoryUsage())

   -- Event is created here
   local event = cutorch.Event()

   cutorch.setStream(2)

   -- assert below will fail without this
   event:waitOn()
   t2:copy(t1View)

   -- revert to default stream
   cutorch.setStream(0)
end

print ("cutorch.hasHalf is ", cutorch.hasHalf)

print('Memory usage before intialization of threads [free memory], [total memory]')
print(cutorch.getMemoryUsage())
threads = Threads(100, function() require 'cutorch'; test_cudaEvent(); end)
print('Memory usage after intialization of threads [free memory], [total memory]')
print(cutorch.getMemoryUsage())
threads:terminate()
print('Memory usage after termination of threads [free memory], [total memory]')
print(cutorch.getMemoryUsage())
