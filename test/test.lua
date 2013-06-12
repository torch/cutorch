require 'cutorch'

local tester
local test = {}
local msize = 100


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
      return diff < tolerance
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
   local x = torch.FloatTensor():rand(msize, 1)
   compareFloatAndCuda(x, 'expand', msize, msize)

   x = torch.FloatTensor():rand(1, msize)
   compareFloatAndCuda(x, 'expand', msize, msize)
end


function test.copyNoncontiguous()
   local x = torch.FloatTensor():rand(msize, 1)
   local f = function(src)
      return src.new(msize, msize):copy(src:expand(msize, msize))
   end
   compareFloatAndCuda(x, f)
end


function test.mean()
   local x = torch.FloatTensor():rand(msize, msize)
   compareFloatAndCuda(x, 'mean')
   compareFloatAndCuda(x, 'mean', 1)
   compareFloatAndCuda(x, 'mean', 2)
end


function test.var()
   local x = torch.FloatTensor():rand(msize, msize)
   compareFloatAndCuda(x, 'var')
   compareFloatAndCuda(x, 'var', 1, true)
   compareFloatAndCuda(x, 'var', 1, false)
   compareFloatAndCuda(x, 'var', 2, true)
   compareFloatAndCuda(x, 'var', 2, false)
end


function test.std()
   local x = torch.FloatTensor():rand(msize, msize)
   compareFloatAndCuda(x, 'std')
   compareFloatAndCuda(x, 'std', 1, true)
   compareFloatAndCuda(x, 'std', 1, false)
   compareFloatAndCuda(x, 'std', 2, true)
   compareFloatAndCuda(x, 'std', 2, false)
end


function cutorch.test()
   math.randomseed(os.time())
   tester = torch.Tester()
   tester:add(test)
   tester:run()
end
