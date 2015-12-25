function torch.CudaTensor.apply(self, func)
   local x = torch.FloatTensor(self:size()):copy(self)
   x:apply(func)
   self:copy(x)
   return self
end

local function Tensor__type(self,type)
   local current = torch.typename(self)
   if not type then return current end
   if type ~= current then
      local new = torch.getmetatable(type).new()
      if self:nElement() > 0 then
         new:resize(self:size()):copy(self)
      end
      return new
   else
      return self
   end
end
local function Tensor__typeAs(self,tensor)
   return self:type(tensor:type())
end

local TensorTypes = {
   float  = 'torch.FloatTensor',
   double = 'torch.DoubleTensor',
   byte   = 'torch.ByteTensor',
   char   = 'torch.CharTensor',
   int    = 'torch.IntTensor',
   short  = 'torch.ShortTensor',
   long   = 'torch.LongTensor',
   cuda       = 'torch.CudaTensor',
   cudaDouble = 'torch.CudaDoubleTensor',
   cudaByte   = 'torch.CudaByteTensor',
   cudaChar   = 'torch.CudaCharTensor',
   cudaInt    = 'torch.CudaIntTensor',
   cudaShort  = 'torch.CudaShortTensor',
   cudaLong   = 'torch.CudaLongTensor'
}


local function Tensor__converter(type)
    return function(self)
        return self:type(type)
    end
end

for _, SrcType in pairs(TensorTypes) do
    for FuncName, DstType in pairs(TensorTypes) do
        rawset(torch.getmetatable(SrcType), FuncName, Tensor__converter(DstType))
    end
end

for _, CudaTensorType in pairs(TensorTypes) do
    rawset(torch.getmetatable(CudaTensorType), 'type', Tensor__type)
    rawset(torch.getmetatable(CudaTensorType), 'typeAs', Tensor__typeAs)
end

do
    local metatable = torch.getmetatable('torch.CudaTensor')
    for _,func in pairs{'expand', 'expandAs', 'view', 'viewAs', 'repeatTensor',
                        'permute', 'split', 'chunk'} do
        rawset(metatable, func, torch[func])
    end
end
