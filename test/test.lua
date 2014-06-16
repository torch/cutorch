require 'cutorch'

local tester
local test = {}
local msize = 100
local minsize = 100
local maxsize = 1000
local nloop = 100
local times = {}


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


local function compareFloatAndCudaTensorArgs(x, fn, ...)
  local x_cpu = x:float()
  local x_cuda = x_cpu:cuda()
  local res_cpu, res_cuda
  -- Transformation of args 
  local tranform_args = function(t, type)
      for k,v in pairs(t) do
        local v_type = torch.Tensor.type(v)
        if v_type == 'torch.FloatTensor' or v_type == 'torch.CudaTensor' or v_type == 'torch.DoubleTensor' then
          t[k] = v:type(type)
        end
      end
    return t
  end
  local cpu_args = tranform_args({...}, 'torch.FloatTensor')
  local cuda_args = tranform_args({...}, 'torch.CudaTensor')
  if type(fn) == 'string' then
      tester:assertne(x_cuda[fn], nil,
         string.format("Missing function CudaTensor.%s", fn))
      res_cpu  = x_cpu[fn](x_cpu, unpack(cpu_args))
      res_cuda = float(x_cuda[fn](x_cuda, unpack(cuda_args)))
   elseif type(fn) == 'function' then
      res_cpu  = fn(x_cpu, unpack(cpu_args))
      res_cuda = float(fn(x_cuda, unpack(cuda_args)))
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

function test.largeNoncontiguous()
  local x = torch.FloatTensor():randn(20, 1, 60, 60)
  local sz = math.floor(torch.uniform(maxsize, 2*maxsize))
  local f = function(src)
    return src.new(20, sz, 60, 60):copy(src:expand(20, sz, 60, 60))
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

function test.index()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local sz3 = math.floor(torch.uniform(10,20))
   local x = torch.FloatTensor():rand(sz1, sz2)

  local longIndex = torch.LongTensor{math.floor(torch.uniform(1, sz1)), math.floor(torch.uniform(1, sz1))}
  local index = 1
  compareFloatAndCuda(x, 'index', index, longIndex)

  index = 2
  longIndex =  torch.LongTensor{math.floor(torch.uniform(1, sz2)), math.floor(torch.uniform(1, sz2))}
  compareFloatAndCuda(x, 'index', index, longIndex)

  x = torch.FloatTensor():rand(sz1)
  index = 1
  longIndex = torch.LongTensor{math.floor(torch.uniform(1, sz1)), math.floor(torch.uniform(1, sz1))}
  compareFloatAndCuda(x, 'index', index, longIndex)

   x = torch.FloatTensor():rand(sz1,sz2,sz3)
   index = 3
   longIndex = torch.randperm(sz3):long()
   compareFloatAndCuda(x, 'index', index, longIndex)
end

function test.indexCopy()
  local sz1 = math.floor(torch.uniform(minsize,maxsize))
  local sz2 = math.floor(torch.uniform(minsize,maxsize))
  local x = torch.FloatTensor():rand(sz1, sz2)

  local longIndex = torch.LongTensor{math.floor(torch.uniform(1, sz1)), math.floor(torch.uniform(1, sz1))}
  local index = 1
  local src = torch.Tensor(2, sz2):uniform(100, 200)
  compareFloatAndCudaTensorArgs(x, 'indexCopy', index, longIndex, src)

  index = 2
  longIndex =  torch.LongTensor{math.floor(torch.uniform(1, sz2)), math.floor(torch.uniform(1, sz2))}
  src = torch.Tensor(sz1, 2):uniform(100, 200):cuda()
  compareFloatAndCudaTensorArgs(x, 'indexCopy', index, longIndex, src)

  x = torch.FloatTensor():rand(sz1)
  index = 1
  longIndex = torch.LongTensor{math.floor(torch.uniform(1, sz1)), math.floor(torch.uniform(1, sz1))}
  src = torch.Tensor(2, sz2):uniform(100, 200)
  compareFloatAndCudaTensorArgs(x, 'indexCopy', index, longIndex, src)

end

function test.indexFill()
  local sz1 = math.floor(torch.uniform(minsize,maxsize))
  local sz2 = math.floor(torch.uniform(minsize,maxsize))
  local x = torch.FloatTensor():rand(sz1, sz2)

  local longIndex = torch.LongTensor{math.floor(torch.uniform(1, sz1)), math.floor(torch.uniform(1, sz1))}
  local index = 1
  local val = torch.randn(1)[1]
  compareFloatAndCuda(x, 'indexFill', index, longIndex, val)

  index = 2
  longIndex =  torch.LongTensor{math.floor(torch.uniform(1, sz2)), math.floor(torch.uniform(1, sz2))}
  val = torch.randn(1)[1]
  compareFloatAndCuda(x, 'indexFill', index, longIndex, val)

  x = torch.FloatTensor():rand(sz1)
  index = 1
  longIndex = torch.LongTensor{math.floor(torch.uniform(1, sz1)), math.floor(torch.uniform(1, sz1))}
  val = torch.randn(1)[1]
  compareFloatAndCuda(x, 'indexFill', index, longIndex, val)

end

function test.renorm()
  local x = torch.randn(10,5):float()
  local maxnorm = x:norm(2,1):mean()
  
  compareFloatAndCuda(x, 'renorm', 2, 2, maxnorm)

  x = torch.randn(3,4,5)
  compareFloatAndCuda(x, 'renorm', 2, 2, maxnorm)
    
  x = torch.randn(3,4,5)
  compareFloatAndCuda(x, 'renorm', 3, 2, maxnorm)
  
  x = torch.randn(3,4,5,100)
  compareFloatAndCuda(x, 'renorm', 3, 2, maxnorm)
  
  x = torch.randn(3,4,5,100)
  compareFloatAndCuda(x, 'renorm', 4, 2, maxnorm)
end

function test.indexSelect()
   --  test for speed
   local n_row = math.random(minsize,maxsize)
   local n_col = math.random(minsize,maxsize)
   local n_idx = math.random(n_col)
   
   local x = torch.randn(n_row, n_col):float()
   local indices = torch.randperm(n_idx):long()
   local z = torch.FloatTensor()
   
   local tm = {}
   local title = string.format('indexSelect ')
   times[title] = tm

   z:index(x, 2, indices)
   local groundtruth = z:clone()
   local clock = torch.Timer()
   for i=1,nloop do
      z:index(x, 2, indices)
   end
   tm.cpu = clock:time().real
   
   x = x:cuda()
   z = torch.CudaTensor()
   
   z:index(x, 2, indices)
   local rescuda = z:clone():float()
   clock:reset()
   for i=1,nloop do
      z:index(x, 2, indices)
   end
   tm.gpu = clock:time().real
   
   tester:assertTensorEq(groundtruth, rescuda, 0.00001, "Error in indexSelect")
end


function cutorch.test(tests)
   math.randomseed(os.time())
   torch.manualSeed(os.time())
   tester = torch.Tester()
   tester:add(test)
   tester:run(tests)
   print ''
   for module,tm in pairs(times) do
      print(module .. ': \t average speedup is ' .. (tm.cpu / (tm.gpu or 1e6)))
   end
end

cutorch.test()
