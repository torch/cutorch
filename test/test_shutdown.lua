local Threads = require 'threads'
require 'cutorch'

local function test_cudaEvent()
   cutorch.reserveStreams(2)
   cutorch.setStream(1)

   local t1 = torch.CudaTensor(10000000):zero()
   local t2 = torch.CudaTensor(1):zero()

   local t1View = t1:narrow(1, 10000000, 1)
   t1:fill(1)

   -- Event is created here
   local event = cutorch.Event()

   cutorch.setStream(2)

   -- assert below will fail without this
   event:waitOn()
   t2:copy(t1View)

   -- revert to default stream
   cutorch.setStream(0)
end

local Gig = 1024*1024*1024

local function test_getMemInfo()
   local sz = Gig*0.1
   local t1 = torch.CudaTensor(sz):zero()
   print('Memory usage after 1st allocation [free memory], [total memory]')
   local free, total = cutorch.getMemoryUsage()
   print(free/Gig, total/Gig)
   local t2 = torch.CudaTensor(sz*1.3):zero()
   print('Memory usage after 2nd allocation [free memory], [total memory]')
   local free, total = cutorch.getMemoryUsage()
   print(free/Gig, total/Gig)
   t1 = nil
   collectgarbage()
   print('Memory usage after 1st deallocation [free memory], [total memory]')
   local free, total = cutorch.getMemoryUsage()
   print(free/Gig, total/Gig)
   t2 = nil
   collectgarbage()
   print('Memory usage after 2nd deallocation [free memory], [total memory]')
   free, total = cutorch.getMemoryUsage()
   print(free/Gig, total/Gig)
end

print ("cutorch.hasHalf is ", cutorch.hasHalf)
print('Memory usage before intialization of threads [free memory], [total memory]')
local free, total = cutorch.getMemoryUsage()
print(free/Gig, total/Gig)
threads = Threads(20, function() require 'cutorch'; test_getMemInfo(); test_cudaEvent(); end)
print('Memory usage after intialization of threads [free memory], [total memory]')
free, total = cutorch.getMemoryUsage()
print(free/Gig, total/Gig)
threads:terminate()
collectgarbage()  
print('Memory usage after termination of threads [free memory], [total memory]')
free, total = cutorch.getMemoryUsage()
print(free/Gig, total/Gig)

