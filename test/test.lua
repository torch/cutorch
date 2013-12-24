require 'cutorch'

local tester
local test = {}
local msize = 100
local minsize = 100
local maxsize = 1000


local function float(x)
   if type(x) == 'number' then
      return x
   else
      return x:float()
   end
end


local function isEqual(a, b, tolerance, ...)
   local diff = a-b
   tolerance = tolerance or 0.000001
   if type(a) == 'number' then
      return math.abs(diff) < tolerance
   else
      return diff:abs():max() < tolerance
   end
end


local function compareFloatAndCuda(x, fn, ...)
   x_cpu    = x:float()
   x_cuda   = x_cpu:cuda()
   local res_cpu, res_cuda
   if type(fn) == 'string' then
      tester:assertne(x_cuda[fn], nil,
         string.format("Missing function CudaTensor.%s", fn))
      res_cpu  = x_cpu[fn](x_cpu, ...)
      res_cuda = float(x_cuda[fn](x_cuda, ...))
   elseif type(fn) == 'function' then
      res_cpu  = fn(x_cpu, ...)
      res_cuda = float(fn(x_cuda, ...))
   else
      error("Incorrect function type")
   end
   local tolerance = 1e-5
   tester:assert(isEqual(res_cpu, res_cuda, tolerance),
      string.format("Divergent results between CPU and CUDA for function '%s'", fn))
end


function test.expand()
   local sz = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz, 1)
   compareFloatAndCuda(x, 'expand', sz, sz)

   x = torch.FloatTensor():rand(1, sz)
   compareFloatAndCuda(x, 'expand', sz, sz)
end


function test.copyNoncontiguous()
   local sz = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz, 1)
   local f = function(src)
      return src.new(sz, sz):copy(src:expand(sz, sz))
   end
   compareFloatAndCuda(x, f)
end


function test.mean()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'mean')
   compareFloatAndCuda(x, 'mean', 1)
   compareFloatAndCuda(x, 'mean', 2)
end


function test.var()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'var')
   -- multi-dim var is not implemented
   -- compareFloatAndCuda(x, 'var', 1, true)
   -- compareFloatAndCuda(x, 'var', 1, false)
   -- compareFloatAndCuda(x, 'var', 2, true)
   -- compareFloatAndCuda(x, 'var', 2, false)
end


function test.std()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'std')
   -- multi-dim std is not implemented
   -- compareFloatAndCuda(x, 'std', 1, true)
   -- compareFloatAndCuda(x, 'std', 1, false)
   -- compareFloatAndCuda(x, 'std', 2, true)
   -- compareFloatAndCuda(x, 'std', 2, false)
end


function cutorch.test()
   math.randomseed(os.time())
   torch.manualSeed(os.time())
   tester = torch.Tester()
   tester:add(test)
   tester:run()
end
