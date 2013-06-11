require 'cutorch'

local tester
local test = {}
local msize = 100


local function compareFloatAndCuda(x, fn, ...)
   x_cpu    = x:float()
   x_cuda   = x_cpu:cuda()
   local res_cpu, res_cuda
   if type(fn) == 'string' then
      tester:assertne(x_cuda[fn], nil,
         string.format("Missing function CudaTensor.%s", fn))
      res_cpu  = x_cpu[fn](x_cpu, ...)
      res_cuda = x_cuda[fn](x_cuda, ...):float()
   elseif type(fn) == 'function' then
      res_cpu  = fn(x_cpu, ...)
      res_cuda = fn(x_cuda, ...):float()
   else
      error("Incorrect function type")
   end
   local tolerance = 1e-5
   tester:assertTensorEq(res_cpu, res_cuda, tolerance,
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


function cutorch.test()
   math.randomseed(os.time())
   tester = torch.Tester()
   tester:add(test)
   tester:run()
end
