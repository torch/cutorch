require "torch"
paths.require("libcutorch")

torch.CudaByteStorage.__tostring__   = torch.ByteStorage.__tostring__
torch.CudaByteTensor.__tostring__    = torch.ByteTensor.__tostring__
torch.CudaCharStorage.__tostring__   = torch.CharStorage.__tostring__
torch.CudaCharTensor.__tostring__    = torch.CharTensor.__tostring__
torch.CudaShortStorage.__tostring__  = torch.ShortStorage.__tostring__
torch.CudaShortTensor.__tostring__   = torch.ShortTensor.__tostring__
torch.CudaIntStorage.__tostring__    = torch.IntStorage.__tostring__
torch.CudaIntTensor.__tostring__     = torch.IntTensor.__tostring__
torch.CudaLongStorage.__tostring__   = torch.LongStorage.__tostring__
torch.CudaLongTensor.__tostring__    = torch.LongTensor.__tostring__
torch.CudaStorage.__tostring__       = torch.FloatStorage.__tostring__
torch.CudaTensor.__tostring__        = torch.FloatTensor.__tostring__
torch.CudaDoubleStorage.__tostring__ = torch.DoubleStorage.__tostring__
torch.CudaDoubleTensor.__tostring__  = torch.DoubleTensor.__tostring__
if cutorch.hasHalf then
   torch.CudaHalfStorage.__tostring__  = torch.FloatStorage.__tostring__
   torch.CudaHalfTensor.__tostring__  = torch.FloatTensor.__tostring__
end

require('cutorch.Tensor')
require('cutorch.FFI')
require('cutorch.test')

local unpack = unpack or table.unpack

function cutorch.withDevice(newDeviceID, closure)
    local curDeviceID = cutorch.getDevice()
    cutorch.setDevice(newDeviceID)
    local vals = {pcall(closure)}
    cutorch.setDevice(curDeviceID)
    if vals[1] then
       return unpack(vals, 2)
    end
    error(unpack(vals, 2))
end

-- Creates a FloatTensor using the CudaHostAllocator.
-- Accepts either a LongStorage or a sequence of numbers.
function cutorch.createCudaHostTensor(...)
   local size
   if not ... then
      size = torch.LongTensor{0}
   elseif torch.isStorage(...) then
      size = torch.LongTensor(...)
   else
      size = torch.LongTensor{...}
   end

   local storage = torch.FloatStorage(cutorch.CudaHostAllocator, size:prod())
   return torch.FloatTensor(storage, 1, size:storage())
end

-- remove this line to disable automatic cutorch heap-tracking
-- for garbage collection
cutorch.setHeapTracking(true)

return cutorch
