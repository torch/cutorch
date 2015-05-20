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
local function Tensor__cuda(self)
   return self:type('torch.CudaTensor')
end
local function Tensor__double(self)
   return self:type('torch.DoubleTensor')
end
local function Tensor__float(self)
   return self:type('torch.FloatTensor')
end

local function Tensor__byte(self)
   return self:type('torch.ByteTensor')
end

local function Tensor__char(self)
   return self:type('torch.CharTensor')
end

local function Tensor__int(self)
   return self:type('torch.IntTensor')
end

local function Tensor__short(self)
   return self:type('torch.ShortTensor')
end

local function Tensor__long(self)
   return self:type('torch.LongTensor')
end

rawset(torch.getmetatable('torch.DoubleTensor'), 'cuda', Tensor__cuda)
rawset(torch.getmetatable('torch.FloatTensor'), 'cuda', Tensor__cuda)
rawset(torch.getmetatable('torch.ByteTensor'), 'cuda', Tensor__cuda)
rawset(torch.getmetatable('torch.CharTensor'), 'cuda', Tensor__cuda)
rawset(torch.getmetatable('torch.IntTensor'), 'cuda', Tensor__cuda)
rawset(torch.getmetatable('torch.ShortTensor'), 'cuda', Tensor__cuda)
rawset(torch.getmetatable('torch.LongTensor'), 'cuda', Tensor__cuda)
rawset(torch.getmetatable('torch.CudaTensor'), 'cuda', Tensor__cuda)

rawset(torch.getmetatable('torch.CudaTensor'), 'type', Tensor__type)
rawset(torch.getmetatable('torch.CudaTensor'), 'typeAs', Tensor__typeAs)
rawset(torch.getmetatable('torch.CudaTensor'), 'double', Tensor__double)
rawset(torch.getmetatable('torch.CudaTensor'), 'float', Tensor__float)
rawset(torch.getmetatable('torch.CudaTensor'), 'byte', Tensor__byte)
rawset(torch.getmetatable('torch.CudaTensor'), 'char', Tensor__char)
rawset(torch.getmetatable('torch.CudaTensor'), 'int', Tensor__int)
rawset(torch.getmetatable('torch.CudaTensor'), 'short', Tensor__short)
rawset(torch.getmetatable('torch.CudaTensor'), 'long', Tensor__long)

do
    local metatable = torch.getmetatable('torch.CudaTensor')
    for _,func in pairs{'expand', 'expandAs', 'view', 'viewAs', 'repeatTensor',
                        'permute', 'split', 'chunk'} do
        rawset(metatable, func, torch[func])
    end
end
