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
   torch.CudaHalfStorage.__tostring__  = torch.HalfStorage.__tostring__
   torch.CudaHalfTensor.__tostring__  = torch.HalfTensor.__tostring__
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

local function longTensorSize(...)
   local size
   if not ... then
      size = torch.LongTensor{0}
   elseif torch.isStorage(...) then
      size = torch.LongTensor(...)
   else
      size = torch.LongTensor{...}
   end
   return size
end

local hostTypes = {'Float', 'Double', 'Int', 'Long', 'Byte'}
if cutorch.hasHalf then
   table.insert(hostTypes, 'Half')
end

for _, ty in ipairs(hostTypes) do
   -- Creates torch Tensors using the CudaHostAllocator.
   -- Accepts either a LongStorage or a sequence of numbers.
   cutorch['createCudaHost' .. ty .. 'Tensor'] = function(...)
      local size = longTensorSize(...)
      local storage = torch[ty .. 'Storage'](cutorch.CudaHostAllocator, size:prod())
      return torch[ty .. 'Tensor'](storage, 1, size:storage())
   end
end

-- Alias to automate creation from both torch and cutorch types
cutorch.createCudaHostTensor = cutorch.createCudaHostFloatTensor

-- Creates a CudaTensor using the CudaUVAAllocator.
-- Accepts either a LongStorage or a sequence of numbers.
local function _createUVATensor(...)
   local size = longTensorSize(...)
   -- See CUDA_C_Programming_guide.pdf for detailed explanation about synchronization
   -- Section J.
   -- "It is worth a comment on the synchronization between host and device. Notice how in
   -- the non-managed example, the synchronous cudaMemcpy() routine is used both to
   -- synchronize the kernel (that is, to wait for it to finish running), and to transfer the data
   -- to the host. The Unified Memory examples do not call cudaMemcpy() and so require an
   -- explicit cudaDeviceSynchronize() before the host program can safely use the output
   -- from the GPU."
   -- Section J.2.2.1.
   -- " Note that if memory is dynamically allocated with cudaMallocManaged() or
   -- cuMemAllocManaged() while the GPU is active, the behavior of the memory is
   -- unspecified until additional work is launched or the GPU is synchronized. Attempting
   -- to access the memory on the CPU during this time may or may not cause a segmentation
   -- fault."
   cutorch.synchronize()
   local storage = torch.FloatStorage(cutorch.CudaUVAAllocator, size:prod())
   return torch.FloatTensor(storage)
end

function cutorch.createFloatUVATensor(...)
   return _createUVATensor(...)
end

-- Creates a CudaTensor using the CudaUVAAllocator.
-- Accepts either a LongStorage or a sequence of numbers.
-- First creates a UVA backed FloatTensor and takes its pointer.
function cutorch.createCudaUVATensor(...)
   -- Delegate actual allocation and synchronization to CPU tensor and
   -- take the pointer.
   local ft = _createUVATensor(...)
   local storage = torch.CudaStorage(
      ft:storage():size(),
      tonumber(torch.data(ft:storage(), true))
   )
   return torch.CudaTensor(storage)
end

-- UVA storage is a single memory location backed by virtual addressing.
-- Converting between CPU / GPU tensor types is done by raw pointer passing.
-- We only support FloatTensor, CudaTensor, Cuda -> float and float -> Cuda atm
function cutorch.toFloatUVATensor(t)
   if not torch.isTensor(t) then
      error('Must use a tensor, got ' .. torch.type(t))
   end
   local storage = torch.FloatStorage(
      t:storage():size(),
      tonumber(torch.data(t:storage(), true))
   )
   assert(cutorch.isManaged(storage))
   return torch.FloatTensor(storage)
end

function cutorch.toCudaUVATensor(t)
   if not torch.isTensor(t) then
      error('Must use a tensor, got ' .. torch.type(t))
   end
   local storage = torch.CudaStorage(
      t:storage():size(),
      tonumber(torch.data(t:storage(), true))
   )
   assert(cutorch.isManaged(storage))
   return torch.CudaTensor(storage)
end

function cutorch.isManaged(t)
   if not torch.isTensor(t) and not torch.isStorage(t) then
      error('Usage: cutorch.isManaged(Tensor|Storage), got ' .. torch.type(t))
   end
   return cutorch.isManagedPtr(tonumber(torch.data(t, true)))
end

-- remove this line to disable automatic cutorch heap-tracking
-- for garbage collection
cutorch.setHeapTracking(true)



function torch.multinomialAliasSetup(probs, state)
   if torch.type(state) == 'table' then 
      state[1], state[2] = torch.multinomialAliasSetup_(probs, state[1], state[2])
   else
      state = {}
      state[1], state[2] = torch.multinomialAliasSetup_(probs)
    end
    return state
 end

function torch.multinomialAlias(output, state)
   torch.CudaTensor.multinomialAlias_(output, state[1], state[2])
   return output
end
return cutorch
