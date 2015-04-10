require "torch"
cutorch = paths.require("libcutorch")

torch.CudaStorage.__tostring__ = torch.FloatStorage.__tostring__
torch.CudaTensor.__tostring__ = torch.FloatTensor.__tostring__

include('Tensor.lua')
include('FFI.lua')
include('test.lua')

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
  local size = torch.LongTensor(torch.isStorage(...) and ... or {...})
  local storage = torch.FloatStorage(cutorch.CudaHostAllocator, size:prod())
  return torch.FloatTensor(storage, 1, size:storage())
end

return cutorch
