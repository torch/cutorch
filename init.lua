require "torch"
cutorch = require "libcutorch"

torch.CudaStorage.__tostring__ = torch.FloatStorage.__tostring__
torch.CudaTensor.__tostring__ = torch.FloatTensor.__tostring__

include('Tensor.lua')
include('FFI.lua')
include('test.lua')

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

return cutorch
