function torch.CudaTensor.apply(self, func)
   local x = torch.FloatTensor(self:size()):copy(self)
   x:apply(func)
   self:copy(x)
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
local function Tensor__cuda(self,type)
   return self:type('torch.CudaTensor')
end
local function Tensor__double(self,type)
   return self:type('torch.DoubleTensor')
end
local function Tensor__float(self,type)
   return self:type('torch.FloatTensor')
end

rawset(torch.getmetatable('torch.DoubleTensor'), 'cuda', Tensor__cuda)
rawset(torch.getmetatable('torch.FloatTensor'), 'cuda', Tensor__cuda)
rawset(torch.getmetatable('torch.CudaTensor'), 'cuda', Tensor__cuda)

rawset(torch.getmetatable('torch.CudaTensor'), 'type', Tensor__type)
rawset(torch.getmetatable('torch.CudaTensor'), 'typeAs', Tensor__typeAs)
rawset(torch.getmetatable('torch.CudaTensor'), 'double', Tensor__double)
rawset(torch.getmetatable('torch.CudaTensor'), 'float', Tensor__float)

for _,func in ipairs({'addmv',
                      'addmm'}) do
      
   local torchfunc = torch.CudaTensor[func]
   torch.CudaTensor[func] = function(self, next1, next2, ...)
                               if type(next1) == 'number' and type(next2) == 'number' then -- beta=next1, alpha=next2
                                  return torchfunc(self, next1, next2, ...)
                               elseif type(next1) == 'number' then -- beta=1, alpha=next1
                                  return torchfunc(self, 1, next1, next2, ...)
                               else -- beta=1, alpha=1
                                  return torchfunc(self, 1, 1, next1, next2, ...)
                               end
                            end      
end

do
    local metatable = torch.getmetatable('torch.CudaTensor')
    for _,func in pairs{'expand', 'expandAs'} do
        rawset(metatable, func, torch[func])
    end
end
