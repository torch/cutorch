local runtests = false
if not cutorch then
   require 'cutorch'
   runtests = true
end

local test = {}
local msize = 100
local minsize = 100
local maxsize = 1000
local minvalue = 2
local maxvalue = 20
local nloop = 100
local times = {}
local test_tolerance = 1e-5
--e.g. unit test cmd: th -lcutorch -e "cutorch.test{'view','viewAs'}"

local function isEqual(a, b, tolerance, ...)
   if a == nil and b == nil then return true end
   if a == nil and b ~= nil then return false end
   if a ~= nil and b == nil then return false end
   if torch.type(b) ~= torch.type(a) then
      b = b:typeAs(a) -- TODO: remove the need for this (a-b doesnt work for bytetensor, cudatensor pairs)
   end
   local diff = a-b
   tolerance = tolerance or 0.000001
   if type(a) == 'number' then
      return math.abs(diff) < tolerance
   else
      if torch.type(diff) ~= 'torch.FloatTensor' then
         diff = diff:float() -- TODO: remove the need for this (byteTensor and abs)
      end
      return diff:abs():max() < tolerance
   end
end

local function compareFloatAndCuda(x, fn, ...)
   local args = {...}
   args['input'] = x
   local x_cpu    = x:float()
   -- keep the size/stride of original tensor
   local x_cuda   = torch.CudaTensor(x_cpu:size(), x_cpu:stride())
   if x_cpu:storage() then
      x_cuda:storage():copy(x_cpu:storage())
   end
   local res1_cpu, res2_cpu, res3_cpu, res4_cpu
   local res1_cuda, res2_cuda, res3_cuda, res4_cuda
   if type(fn) == 'string' then
      tester:assertne(x_cuda[fn], nil,
         string.format("Missing function CudaTensor.%s", fn))
      res1_cpu, res2_cpu, res3_cpu, res4_cpu  = x_cpu[fn](x_cpu, ...)
      res1_cuda, res2_cuda, res3_cuda, res4_cuda = x_cuda[fn](x_cuda, ...)
   elseif type(fn) == 'function' then
      res1_cpu, res2_cpu, res3_cpu, res4_cpu  = fn(x_cpu, ...)
      res1_cuda, res2_cuda, res3_cuda, res4_cuda = fn(x_cuda, ...)
   else
      error("Incorrect function type")
   end
   local tolerance = test_tolerance
   if not isEqual(res1_cpu, res1_cuda, tolerance) then
      print(args)
      tester:assert(false,
                    string.format("Divergent results between CPU and CUDA" ..
                                  " for function '%s' (return value 1)", tostring(fn)))
   end
   if not isEqual(res2_cpu, res2_cuda, tolerance) then
      print(args)
      tester:assert(false,
                    string.format("Divergent results between CPU and CUDA" ..
                                  " for function '%s' (return value 2)", tostring(fn)))
   end
   if not isEqual(res3_cpu, res3_cuda, tolerance) then
      print(args)
      tester:assert(false,
                    string.format("Divergent results between CPU and CUDA" ..
                                  " for function '%s' (return value 3)", tostring(fn)))
   end
   if not isEqual(res4_cpu, res4_cuda, tolerance) then
      print(args)
      tester:assert(false,
                    string.format("Divergent results between CPU and CUDA" ..
                                  " for function '%s' (return value 4)", tostring(fn)))
   end
end

local function compareFloatAndCudaTensorArgs(x, fn, ...)
   local x_cpu = x:float()
   -- keep the size/stride of original tensor
   local x_cuda = torch.CudaTensor(x_cpu:size(), x_cpu:stride())
   if x_cpu:storage() then
      x_cuda:storage():copy(x_cpu:storage())
   end
   local res1_cpu, res2_cpu, res3_cpu, res4_cpu
   local res1_cuda, res2_cuda, res3_cuda, res4_cuda
   -- Transformation of args
   local tranform_args = function(t, type)
      for k,v in pairs(t) do
         local v_type = torch.Tensor.type(v)
         if v_type == 'torch.FloatTensor' or v_type == 'torch.CudaTensor' or v_type == 'torch.DoubleTensor' then
            t[k] = v:type(type).new(v:size(), v:stride())
            if v:storage() then t[k]:storage():copy(v:storage()) end
         end
      end
      return t
   end
   local cpu_args = tranform_args({...}, 'torch.FloatTensor')
   local cuda_args = tranform_args({...}, 'torch.CudaTensor')
   if type(fn) == 'string' then
      tester:assertne(x_cuda[fn], nil,
         string.format("Missing function CudaTensor.%s", fn))
      res1_cpu, res2_cpu, res3_cpu, res4_cpu  = x_cpu[fn](x_cpu, unpack(cpu_args))
      res1_cuda, res2_cuda, res3_cuda, res4_cuda = x_cuda[fn](x_cuda, unpack(cuda_args))
   elseif type(fn) == 'function' then
      res1_cpu, res2_cpu, res3_cpu, res4_cpu  = fn(x_cpu, unpack(cpu_args))
      res1_cuda, res2_cuda, res3_cuda, res4_cuda = fn(x_cuda, unpack(cuda_args))
   else
      error("Incorrect function type")
   end
   local tolerance = test_tolerance
   tester:assert(isEqual(res1_cpu, res1_cuda, tolerance),
                 string.format("Divergent results between CPU and CUDA for function '%s'", fn))
   tester:assert(isEqual(res2_cpu, res2_cuda, tolerance),
                 string.format("Divergent results between CPU and CUDA for function '%s'", fn))
   tester:assert(isEqual(res3_cpu, res3_cuda, tolerance),
                 string.format("Divergent results between CPU and CUDA for function '%s'", fn))
   tester:assert(isEqual(res4_cpu, res4_cuda, tolerance),
                 string.format("Divergent results between CPU and CUDA for function '%s'", fn))
end

function test.squeeze()
   local sz = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz, 1, sz, 1)
   compareFloatAndCuda(x, 'squeeze')

   local y = x:cuda():squeeze()
   tester:assert(y:dim() == 2, "squeeze err")

   x = torch.FloatTensor():rand(sz, 1, 1, sz)
   compareFloatAndCuda(x, 'squeeze', 2)

   local y = x:cuda():squeeze(2)
   tester:assert(y:dim() == 3, "squeeze1d err")
end

function test.expand()
   local sz = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz, 1)
   compareFloatAndCuda(x, 'expand', sz, sz)

   x = torch.FloatTensor():rand(1, sz)
   compareFloatAndCuda(x, 'expand', sz, sz)
end

function test.view()
   local sz = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz, 3)
   compareFloatAndCuda(x, 'view', sz, 3, 1)
end

function test.viewAs()
   local sz = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz, 3)
   local y = torch.FloatTensor():rand(sz, 3, 1)
   compareFloatAndCudaTensorArgs(x, 'viewAs', y)
end

function test.repeatTensor()
   local sz = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz, 3)
   compareFloatAndCuda(x, 'repeatTensor', sz, 2)
end

function test.permute()
   local perm = torch.randperm(7):totable()
   local x = torch.FloatTensor():rand(1, 2, 3, 4, 5, 6, 7)
   compareFloatAndCuda(x, 'permute', unpack(perm))
end

function test.copyRandomizedTest()
   local maxSize = 1000000 -- 1M elements max
   local ndimInput = torch.random(10)
   local function randomSizeGenerator(ndimInput)
      local size = {}
      local totalSize = 1
      for i = 1, ndimInput do
         size[i] = torch.random(25)
         totalSize = totalSize * size[i]
      end
      return size, totalSize
   end
   local inputSize, nElem = randomSizeGenerator(ndimInput)
   local attemptsAtSizeGeneration = 1
   while nElem > maxSize do
      attemptsAtSizeGeneration = attemptsAtSizeGeneration + 1
      -- make atmost 100 attempts to generate sizes randomly.
      -- this guarantees that even in the worst case,
      -- this test does not run forever
      if attemptsAtSizeGeneration == 100 then
         inputSize = {1, 10, 100}
         break
      end
      inputSize, nElem = randomSizeGenerator(ndimInput)
   end

   -- http://rosettacode.org/wiki/Prime_decomposition#Lua
   local function IsPrime(n)
      if n <= 1 or (n ~= 2 and n % 2 == 0) then return false end
      for i = 3, math.sqrt(n), 2 do if n % i == 0 then return false end end
         return true
   end
   local function PrimeDecomposition(n)
      local f = {}
      if IsPrime(n) then f[1] = n; return f end
      local i = 2
      repeat
         while n % i == 0 do f[#f + 1] = i; n = n / i end
         repeat i = i + 1 until IsPrime( i )
      until n == 1
      return f
   end
   local function constructOutput(size)
      local outputSize = {}
      for i = 1, #size do outputSize[i] = size[i] end
      for i = 1, 10 do -- 10 randomizations
         -- pick an input dim
         local dim = torch.random(1, #size)
         -- factor it
         local factors = PrimeDecomposition(outputSize[dim])
         if #factors ~= 0 then
            -- remove one of the factors
            local factor = factors[torch.random(#factors)]
            local addNewDim = torch.random(1, 2)
            if addNewDim == 1 then -- add it as a new dimension
               outputSize[dim] = outputSize[dim] / factor
               -- where to insert new dimension
               local where = torch.random(1, #outputSize)
               local o = {}
               o[where] = factor
               local index = 1
               for j = 1, #outputSize + 1 do
                  if j == where then
                     o[j] = factor
                  else
                     o[j] = outputSize[index]
                     index = index + 1
                  end
               end
               outputSize = o
            else -- or multiply the factor to another dimension
               local where = torch.random(1, #outputSize)
               outputSize[dim] = outputSize[dim] / factor
               outputSize[where] = outputSize[where] * factor
            end
         end
      end
      return outputSize
   end
   local outputSize = constructOutput(inputSize)
   local nelem1 = 1
   local nelem2 = 1
   for i = 1, #inputSize do nelem1 = nelem1 * inputSize[i] end
   for i = 1, #outputSize do nelem2 = nelem2 * outputSize[i] end
   tester:assert(nelem1, nelem2, 'input and output sizes have to be the same')
   local input, output

   local function createHoledTensor(size)
      local osize = {}
      for i = 1, #size do osize[i] = size[i] end
      -- randomly inflate a few dimensions in osize
      for i = 1, 3 do
         local dim = torch.random(1,#osize)
         local add = torch.random(4, 15)
         osize[dim] = osize[dim] + add
      end
      local input = torch.FloatTensor(torch.LongStorage(osize))
      -- now extract the input of correct size from 'input'
      for i = 1, #size do
         if input:size(i) ~= size[i] then
            local bounds = torch.random(1, input:size(i) - size[i] + 1)
            input = input:narrow(i, bounds, size[i])
         end
      end
      return input
   end

   -- extract a sub-cube with probability 50%
   -- (to introduce unreachable storage locations)
   local holedInput = torch.random(1, 2)
   local holedOutput = torch.random(1, 2)
   if holedInput == 1 then
      input = createHoledTensor(inputSize)
   else
      input = torch.FloatTensor(torch.LongStorage(inputSize))
   end
   input:storage():fill(-150)
   input:copy(torch.linspace(1, input:nElement(), input:nElement()))

   if holedOutput == 1 then
      output = createHoledTensor(outputSize)
   else
      output = torch.FloatTensor(torch.LongStorage(outputSize))
   end

   output:storage():fill(-100)
   output:fill(-1)
   -- function to randomly transpose a tensor
   local function randomlyTranspose(input)
      local d1 = torch.random(1, input:dim())
      local d2 = torch.random(1, input:dim())
      if d1 ~= d2 then input = input:transpose(d1, d2) end
      return input
   end
   -- randomly transpose with 50% prob
   local transposeInput = torch.random(1, 2)
   local transposeOutput = torch.random(1, 2)
   if transposeInput == 1 then
      for i = 1, 10 do input = randomlyTranspose(input) end
   end
   if transposeOutput == 1 then
      for i = 1, 10 do output = randomlyTranspose(output) end
   end

   local input_tensor_float = input
   local output_tensor_float = output
   local input_storage_float = input:storage()
   local output_storage_float = output:storage()
   local input_storage_cuda =
      torch.CudaStorage(input_storage_float:size()):copy(input_storage_float)
   local output_storage_cuda =
      torch.CudaStorage(output_storage_float:size()):copy(output_storage_float)
   local input_tensor_cuda = torch.CudaTensor(input_storage_cuda,
                                          input_tensor_float:storageOffset(),
                                          input_tensor_float:size(),
                                          input_tensor_float:stride())
   local output_tensor_cuda = torch.CudaTensor(output_storage_cuda,
                                          output_tensor_float:storageOffset(),
                                          output_tensor_float:size(),
                                          output_tensor_float:stride())

   output_tensor_float:copy(input_tensor_float)
   output_tensor_cuda:copy(input_tensor_cuda)

   -- now compare output_storage_cuda and output_storage_float for exactness
   local flat_tensor_float = torch.FloatTensor(input_storage_float)
   local flat_storage_cuda =
      torch.FloatStorage(input_storage_cuda:size()):copy(input_storage_cuda)
   local flat_tensor_cuda = torch.FloatTensor(flat_storage_cuda)

   local err = (flat_tensor_float - flat_tensor_cuda):abs():max()
   if err ~= 0 then
      print('copyRandomizedTest failure input size: ', input:size())
      print('copyRandomizedTest failure input stride: ', input:stride())
      print('copyRandomizedTest failure output size: ', output:size())
      print('copyRandomizedTest failure output stride: ', output:stride())
   end
   tester:assert(err == 0, 'diverging input and output in copy test')
end

function test.copyNoncontiguous()
     local x = torch.FloatTensor():rand(1, 1)
     local f = function(src)
        return src.new(2, 2):copy(src:expand(2, 2))
     end
     compareFloatAndCuda(x, f)

   local sz = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz, 1)
   local f = function(src)
      return src.new(sz, sz):copy(src:expand(sz, sz))
   end
   compareFloatAndCuda(x, f)

   x = torch.FloatTensor():rand(sz, sz, 2)
   local f = function(src)
      return src.new(sz, sz):copy(src[{{},{},{2}}])
   end
   compareFloatAndCuda(x, f)

   x = torch.FloatTensor():rand(2, sz, sz)
   local f = function(src)
      return src.new(sz, sz):copy(src[{{2},{},{}}])
   end
   compareFloatAndCuda(x, f)

   x = torch.FloatTensor():rand(sz, 2, sz)
   local f = function(src)
      return src.new(sz, sz):copy(src[{{},{2},{}}])
   end
   compareFloatAndCuda(x, f)

   x = torch.FloatTensor():rand(sz, 2, sz)
   local f = function(src)
      return src.new(sz, 1, sz):copy(src[{{},{2},{}}])
   end
   compareFloatAndCuda(x, f)

   x = torch.FloatTensor():rand(sz, sz):transpose(1,2)
   local f = function(src)
      return src.new(sz, sz):copy(src)
   end
   compareFloatAndCuda(x, f)

   -- case for https://github.com/torch/cutorch/issues/90
   do
      local val = 1
      local ps = torch.LongStorage({4, 4, 4})
      local cube = torch.Tensor(ps):apply(
         function()
            val = val + 1
            return val
         end
                                     ):cuda()

      local ps = torch.LongStorage({4, 12})
      local x = torch.CudaTensor(ps):fill(-1)

      local l = 2
      local h = 1
      local w = 2

      x[{{1},{1,9}}]:copy(cube[l][{{h,h+2},{w,w+2}}])
      tester:assert((x[{1,{1,9}}]-cube[l][{{h,h+2},{w,w+2}}]):abs():max() == 0,
         'diverging input and output in copy test')
   end
end

function test.largeNoncontiguous()
   local x = torch.FloatTensor():randn(20, 1, 60, 60)
   local sz = math.floor(torch.uniform(maxsize, 2*maxsize))
   local f = function(src)
      return src.new(20, sz, 60, 60):copy(src:expand(20, sz, 60, 60))
   end
   compareFloatAndCuda(x, f)
end

function test.zero()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'zero')
end

function test.fill()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local v = torch.uniform()
   compareFloatAndCudaTensorArgs(x, 'fill', v)
end

function test.reshape()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))*2
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'reshape', sz1/2, sz2*2)
end

function test.zeros()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local t = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.CudaTensor')
   local x = torch.zeros(sz1, sz2)
   assert(x:sum() == 0)
   torch.setdefaulttensortype(t)
end

function test.ones()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local t = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.CudaTensor')
   local x = torch.ones(sz1, sz2)
   assert(x:sum() == x:nElement())
   torch.setdefaulttensortype(t)
end


function test.add()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor():rand(sz1, sz2)
   local v = torch.uniform()
   compareFloatAndCudaTensorArgs(x, 'add', z)
   compareFloatAndCudaTensorArgs(x, 'add', z, v)
   compareFloatAndCudaTensorArgs(x, 'add', y, z)
   compareFloatAndCudaTensorArgs(x, 'add', y, v, z)
end

function test.cmul()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'cmul', y)
end

function test.cpow()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'cpow', y)
end

function test.cdiv()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'cdiv', y)
end

function test.cdiv3()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor(sz1, sz2)
   compareFloatAndCudaTensorArgs(z, 'cdiv', x, y)
end

function test.addcmul()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'addcmul', y, z)
   compareFloatAndCudaTensorArgs(x, 'addcmul', torch.uniform(), y, z)

   local r = torch.zeros(sz1, sz2)
   compareFloatAndCudaTensorArgs(r, 'addcmul', x, y, z)
   compareFloatAndCudaTensorArgs(r, 'addcmul', x, torch.uniform(), y, z)
end

function test.addcdiv()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'addcdiv', y, z)
   compareFloatAndCudaTensorArgs(x, 'addcdiv', torch.uniform(), y, z)

   local r = torch.zeros(sz1, sz2)
   compareFloatAndCudaTensorArgs(r, 'addcdiv', x, y, z)
   compareFloatAndCudaTensorArgs(r, 'addcdiv', x, torch.uniform(), y, z)
end

function test.logicalValue()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'gt', y, 0.3)
   compareFloatAndCuda(x, 'gt', 0.3)
end

function test.logicalTensor()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'gt', z)
   compareFloatAndCudaTensorArgs(x, 'gt', y, z)
end

function test.mean()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'mean')
   compareFloatAndCuda(x, 'mean', 1)
   compareFloatAndCuda(x, 'mean', 2)
end

function test.max()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.randperm(sz1 * sz2):view(sz1, sz2):float()
   compareFloatAndCuda(x, 'max')
   compareFloatAndCuda(x, 'max', 1)
   compareFloatAndCuda(x, 'max', 2)
end

function test.min()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.randperm(sz1 * sz2):view(sz1, sz2):float()
   compareFloatAndCuda(x, 'min')
   compareFloatAndCuda(x, 'min', 1)
   compareFloatAndCuda(x, 'min', 2)
end

function test.sum()
   local minsize = 10
   local maxsize = 20
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   test_tolerance = 1e-1
   compareFloatAndCuda(x, 'sum')
   compareFloatAndCuda(x, 'sum', 1)
   compareFloatAndCuda(x, 'sum', 2)
   test_tolerance = 1e-5
end

function test.cumsum()
   local minsize = 10
   local maxsize = 20
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'cumsum')
   compareFloatAndCuda(x, 'cumsum', 1)
   compareFloatAndCuda(x, 'cumsum', 2)
end

function test.prod()
   local minsize = 10
   local maxsize = 20
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'prod')
   compareFloatAndCuda(x, 'prod', 1)
   compareFloatAndCuda(x, 'prod', 2)
end

function test.cumprod()
   local minsize = 10
   local maxsize = 20
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'cumprod')
   compareFloatAndCuda(x, 'cumprod', 1)
   compareFloatAndCuda(x, 'cumprod', 2)
end

function test.round()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'round')
end

function test.var()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'var')
   compareFloatAndCuda(x, 'var', 1, true)
   compareFloatAndCuda(x, 'var', 1, false)
   compareFloatAndCuda(x, 'var', 2, true)
   compareFloatAndCuda(x, 'var', 2, false)
end

function test.std()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'std')
   compareFloatAndCuda(x, 'std', 1, true)
   compareFloatAndCuda(x, 'std', 1, false)
   compareFloatAndCuda(x, 'std', 2, true)
   compareFloatAndCuda(x, 'std', 2, false)
end

-- Test element-wise unary operators with both one and two arguments.
local function testUnary1(fn)
   local function test()
      local sz1 = math.floor(torch.uniform(minsize,maxsize))
      local sz2 = math.floor(torch.uniform(minsize,maxsize))
      local x = torch.FloatTensor():rand(sz1, sz2)
      compareFloatAndCudaTensorArgs(x, fn)
   end
   return test
end

local function testUnary2(fn)
   local function test()
      local sz1 = math.floor(torch.uniform(minsize,maxsize))
      local sz2 = math.floor(torch.uniform(minsize,maxsize))
      local x = torch.FloatTensor():rand(sz1, sz2)
      local y = torch.FloatTensor()
      compareFloatAndCudaTensorArgs(y, fn, x)
   end
   return test
end

for _,name in ipairs({"log", "log1p", "exp",
                      "cos", "acos", "cosh",
                      "sin", "asin", "sinh",
                      "tan", "atan", "tanh",
                      "sqrt",
                      "ceil", "floor",
                      "abs", "sign"}) do

   test[name .. "1"] = testUnary1(name)
   test[name .. "2"] = testUnary2(name)

end

function test.atan2(fn)
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor()
   compareFloatAndCudaTensorArgs(z, 'atan2', x, y)
end

function test.pow1()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local pow = torch.uniform(minvalue,maxvalue)
   compareFloatAndCudaTensorArgs(x, 'pow', pow)
end

function test.pow2()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor()
   local pow = torch.uniform(minvalue,maxvalue)
   compareFloatAndCudaTensorArgs(y, 'pow', x, pow)
end

function test.powExponentTensor()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local pow = torch.uniform(minvalue,maxvalue)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor()
   compareFloatAndCudaTensorArgs(y, 'pow', pow, x)
end

function test.clamp1()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2):mul(5):add(-2.5)
   local min_val = -1
   local max_val = 1
   x[1][1] = min_val - 1
   if sz2 >= 2 then
     x[1][2] = max_val + 1
   end
   compareFloatAndCudaTensorArgs(x, 'clamp', min_val, max_val)
end

function test.clamp2()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local x = torch.FloatTensor():rand(sz1, sz2):mul(5):add(-2.5)
   local min_val = -1
   local max_val = 1
   x[1][1] = min_val - 1
   if sz2 >= 2 then
     x[1][2] = max_val + 1
   end
   local y = torch.FloatTensor():resizeAs(x)
   compareFloatAndCudaTensorArgs(y, 'clamp', x, min_val, max_val)
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
   local sz1 = math.floor(torch.uniform(minsize,maxsize)) -- dim1
   local sz2 = math.floor(torch.uniform(minsize,maxsize)) -- dim2
   local x = torch.FloatTensor():rand(sz1, sz2) -- input


   -- Case 1: 2D tensor, indexCopy over first dimension, 2 indices
   -- choose two indices from the first dimension, i.e. [1,sz1]
   local longIndex = torch.LongTensor{math.floor(torch.uniform(1, sz1)), math.floor(torch.uniform(1, sz1))}
   local index = 1
   local src = torch.Tensor(2, sz2):uniform()
   compareFloatAndCudaTensorArgs(x, 'indexCopy', index, longIndex, src)

   -- Case 2: 2D tensor, indexCopy over second dimension, 2 indices
   index = 2
   longIndex =  torch.LongTensor{math.floor(torch.uniform(1, sz2)), math.floor(torch.uniform(1, sz2))}
   src = torch.Tensor(sz1, 2):uniform():cuda()
   compareFloatAndCudaTensorArgs(x, 'indexCopy', index, longIndex, src)

   -- Case 3: 1D tensor, indexCopy over 1st dimension, 2 indices
   x = torch.FloatTensor():rand(sz1)
   index = 1
   longIndex = torch.LongTensor{math.floor(torch.uniform(1, sz1)), math.floor(torch.uniform(1, sz1))}
   src = torch.Tensor(2):uniform()
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

function test.addmv()
   --[[ Size ]]--
   local sizes = {
      {2,1},
      {1,2},
      {1,1},
      {3,4},
      {3,3},
      {15,18},
      {19,15}
   }
   for _, size in pairs(sizes) do
      local n, m = unpack(size)
      local c = torch.zeros(n)
      local a = torch.randn(n, m)
      local b = torch.randn(m)
      compareFloatAndCudaTensorArgs(c, 'addmv', torch.normal(), torch.normal(), a, b)
   end
end

function test.mv()
   --[[ Size ]]--
   local sizes = {
      {2,1},
      {1,2},
      {1,1},
      {3,4},
      {3,3},
      {15,18},
      {19,15}
   }
   for _, size in pairs(sizes) do
      local n, m = unpack(size)
      local c = torch.zeros(n)
      local a = torch.randn(n, m)
      local b = torch.randn(m)
      compareFloatAndCudaTensorArgs(c, 'mv', a, b)
   end
end

function test.addr()
   --[[ Size ]]--
   local sizes = {
      {2,1},
      {1,2},
      {1,1},
      {3,4},
      {3,3},
      {15,18},
      {19,15}
   }
   for _, size in pairs(sizes) do
      local n, m = unpack(size)
      local c = torch.zeros(n,m)
      local a = torch.randn(n)
      local b = torch.randn(m)
      compareFloatAndCudaTensorArgs(c, 'addr', torch.normal(), a, b)
   end
end

function test.addmm()
   --[[ Size ]]--
   local sizes = {
      {16, 3, 1},
      {1, 12, 1},
      {24, 23, 22},
      {1, 1, 1},
      {1, 1, 7},
      {12, 1, 12},
      {10, 10, 10},
   }
   for _, size in pairs(sizes) do
      local n, k, m = unpack(size)
      local c = torch.zeros(n, m)
      local a = torch.randn(n, k)
      local b = torch.randn(k, m)
      compareFloatAndCudaTensorArgs(c, 'addmm', torch.normal(), torch.normal(), a, b)
   end
end

function test.mm()
   --[[ Size ]]--
   local sizes = {
      {16, 3, 1},
      {1, 12, 1},
      {24, 23, 22},
      {1, 1, 1},
      {1, 1, 7},
      {12, 1, 12},
      {10, 10, 10},
   }
   for _, size in pairs(sizes) do
      local n, k, m = unpack(size)
      local c = torch.zeros(n, m)
      local a = torch.randn(n, k)
      local b = torch.randn(k, m)
      compareFloatAndCudaTensorArgs(c, 'mm', a, b)
   end
end

function test.baddbmm()
   local sizes = {
      {16, 3, 1, 4},
      {1, 12, 1, 7},
      {24, 23, 22, 21},
      {1, 1, 1, 1},
      {1, 1, 7, 4},
      {12, 1, 12, 1},
      {10, 10, 10, 10},
   }
   for _, size in pairs(sizes) do
      local b, n, k, m = unpack(size)
      local cs = torch.randn(b, n, m)
      local as = torch.randn(b, n, k)
      local bs = torch.randn(b, k, m)
      compareFloatAndCudaTensorArgs(cs, 'baddbmm', as, bs)
   end
end

function test.baddbmmTransposed()
   local b, n, k, m = 16, 3, 8, 4
   -- Can't use compareFloatAndCudaTensorArgs because the transposition will be
   -- lost when converting the tensor to a CudaTensor.
   local c_cpu = torch.randn(m, n, b)  -- First and last dimensions will be tranposed.
   local a_cpu = torch.randn(n, b, k)  -- First two dimensions will be transposed.
   local b_cpu = torch.randn(b, m, k)  -- Last two dimensions will be transposed.

   local c_cuda = c_cpu:cuda()
   local a_cuda = a_cpu:cuda()
   local b_cuda = b_cpu:cuda()

   c_cpu = c_cpu:transpose(1, 3)
   c_cuda = c_cuda:transpose(1, 3)
   a_cpu = a_cpu:transpose(1, 2)
   a_cuda = a_cuda:transpose(1, 2)
   b_cpu = b_cpu:transpose(2, 3)
   b_cuda = b_cuda:transpose(2, 3)

   c_cpu:baddbmm(a_cpu, b_cpu)
   c_cuda:baddbmm(a_cuda, b_cuda)

   tester:assert(isEqual(c_cpu, c_cuda, 1e-5),
                 string.format("Divergent results between CPU and CUDA for function 'bmm'"))
end

function test.bmm()
   local sizes = {
      {16, 3, 1, 4},
      {1, 12, 1, 7},
      {24, 23, 22, 21},
      {1, 1, 1, 1},
      {1, 1, 7, 4},
      {12, 1, 12, 1},
      {10, 10, 10, 10},
   }
   for _, size in pairs(sizes) do
      local b, n, k, m = unpack(size)
      local cs = torch.zeros(b, n, m)
      local as = torch.randn(b, n, k)
      local bs = torch.randn(b, k, m)
      compareFloatAndCudaTensorArgs(cs, 'bmm', as, bs)
   end
end

function test.bmmTransposed()
   local b, n, k, m = 16, 3, 8, 4
   -- Can't use compareFloatAndCudaTensorArgs because the transposition will be
   -- lost when converting the tensor to a CudaTensor.
   local c_cpu = torch.zeros(b, n, m)
   local a_cpu = torch.randn(b, k, n)  -- Last two dimensions will be transposed.
   local b_cpu = torch.randn(m, k, b)  -- First and last dimensions will be transposed.

   local c_cuda = c_cpu:cuda()
   local a_cuda = a_cpu:cuda()
   local b_cuda = b_cpu:cuda()

   a_cpu = a_cpu:transpose(2, 3)
   a_cuda = a_cuda:transpose(2, 3)
   b_cpu = b_cpu:transpose(1, 3)
   b_cuda = b_cuda:transpose(1, 3)

   c_cpu:bmm(a_cpu, b_cpu)
   c_cuda:bmm(a_cuda, b_cuda)

   tester:assert(isEqual(c_cpu, c_cuda, 1e-5),
                 string.format("Divergent results between CPU and CUDA for function 'bmm'"))
end

function test.ger()
   --[[ Size ]]--
   local sizes = {
      {16, 1},
      {1, 12},
      {24, 23},
      {1, 1},
      {33, 7},
      {12, 14},
      {10, 10},
   }
   for _, size in pairs(sizes) do
      local n, m = unpack(size)
      local c = torch.zeros(n, m)
      local a = torch.randn(n)
      local b = torch.randn(m)
      compareFloatAndCudaTensorArgs(c, 'ger', a, b)
   end
end

function test.isSameSizeAs()
   local t1 = torch.CudaTensor(3, 4, 9, 10)
   local t2 = torch.CudaTensor(3, 4)
   local t3 = torch.CudaTensor(1, 9, 3, 3)
   local t4 = torch.CudaTensor(3, 4, 9, 10)

   tester:assert(t1:isSameSizeAs(t2) == false, "wrong answer ")
   tester:assert(t1:isSameSizeAs(t3) == false, "wrong answer ")
   tester:assert(t1:isSameSizeAs(t4) == true, "wrong answer ")
end

-- Test random number generation.
local function checkIfUniformlyDistributed(t, min, max)
   tester:assertge(t:min(), min - 1e-6, "values are too low")
   tester:assertle(t:max(), max + 1e-6, "values are too high")
   tester:assertalmosteq(t:mean(), (min + max) / 2, 0.1, "mean is wrong")
end

function test.uniform()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local min = torch.uniform()
   local max = min + torch.uniform()
   local t = torch.CudaTensor(sz1, sz2)

   t:uniform(min, max)
   checkIfUniformlyDistributed(t, min, max)
end

function test.bernoulli()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local p = torch.uniform()
   local t = torch.CudaTensor(sz1, sz2)

   t:bernoulli(p)
   tester:assertalmosteq(t:mean(), p, 0.1, "mean is not equal to p")
   local f = t:float()
   tester:assertTensorEq(f:eq(1):add(f:eq(0)):float(),
                         torch.FloatTensor(sz1, sz2):fill(1),
                         1e-6,
                         "each value must be either 0 or 1")
end

function test.normal()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local mean, std = torch.uniform(), torch.uniform()
   local tolerance = 0.01
   local t = torch.CudaTensor(sz1, sz2)

   t:normal(mean, std)
   tester:assertalmosteq(t:mean(), mean, tolerance, "mean is wrong")
   tester:assertalmosteq(t:std(), std, tolerance, "standard deviation is wrong")
end

function test.logNormal()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local mean, std = torch.uniform(), torch.uniform()
   local tolerance = 0.01
   local t = torch.CudaTensor(sz1, sz2)

   t:logNormal(mean, std)
   local logt = t:log()
   tester:assertalmosteq(logt:mean(), mean, tolerance, "mean is wrong")
   tester:assertalmosteq(logt:std(), std, tolerance, "standard deviation is wrong")
end

function test.geometric()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local p = torch.uniform()
   local t = torch.CudaTensor(sz1, sz2)

   t:geometric(p)
   local u = torch.FloatTensor(sz1, sz2):fill(1) -
                 ((t:float() - 1) * math.log(p)):exp()
   checkIfUniformlyDistributed(u, 0, 1)
end

function test.exponential()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local lambda = torch.uniform()
   local t = torch.CudaTensor(sz1, sz2)

   t:exponential(lambda)
   local u = torch.FloatTensor(sz1, sz2):fill(1) -
                 (t:float() * -lambda):exp()
   checkIfUniformlyDistributed(u, 0, 1)
end

function test.cauchy()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local median, sigma = torch.uniform(), torch.uniform()
   local t = torch.CudaTensor(sz1, sz2)

   t:cauchy(median, sigma)
   local u = ((t:float() - median) / sigma):atan() / math.pi + 0.5
   checkIfUniformlyDistributed(u, 0, 1)
end

function test.random_seed()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local mean, std = torch.uniform(), torch.uniform()
   local tolerance = 0.01
   local t = torch.CudaTensor(sz1, sz2)
   local u = torch.CudaTensor(sz1, sz2)

   local seed = cutorch.seed()
   t:normal(mean, std)
   cutorch.manualSeed(seed)
   u:normal(mean, std)
   tester:assertTensorEq(t:float(), u:float(), 1e-6, "values not equal after resetting the seed")
end

function test.restore_rng()
   local sz1 = math.floor(torch.uniform(minsize,maxsize))
   local sz2 = math.floor(torch.uniform(minsize,maxsize))
   local mean, std = torch.uniform(), torch.uniform()
   local tolerance = 0.01
   local t = torch.CudaTensor(sz1, sz2)
   local u = torch.CudaTensor(sz1, sz2)

   local seed = cutorch.seed()
   local rng = cutorch.getRNGState()
   t:normal(mean, std)
   -- Change the seed so we can check that restoring the RNG state also restores the seed.
   cutorch.manualSeed(seed + 123)
   cutorch.setRNGState(rng)
   u:normal(mean, std)
   tester:assertTensorEq(t:float(), u:float(), 1e-6, "values not equal after restoring the RNG state")
   tester:asserteq(cutorch.initialSeed(), seed, "seed was not restored")
end

function test.multi_gpu_random()
   local rs = cutorch.getRNGState()
   cutorch.manualSeedAll(1) -- set all device seeds to be the same

   -- requires at least 2 devices
   local device_count = cutorch.getDeviceCount()
   if device_count < 2 then
      return
   end
   cutorch.setDevice(1)
   local n = 3
   local expected = torch.CudaTensor(n):uniform():float()
   for i = 2, device_count do
      cutorch.setDevice(i)
      local actual = torch.CudaTensor(n):uniform():float()
      tester:assert(isEqual(expected, actual), "random tensors dont seem to be equal")
   end
   cutorch.setRNGState(rs) -- cleanup after yourself
end

function test.get_device()
    local device_count = cutorch.getDeviceCount()
    local tensors = { }
    for i = 1,device_count do
        table.insert(tensors, torch.Tensor():cuda())
    end
    -- Unallocated tensors are on device 0
    for i = 1,device_count do
       tester:assert(tensors[i]:getDevice() == 0, "unallocated tensor does not have deviceID 0")
       -- Now allocate it
       cutorch.setDevice(i)
       tensors[i]:resize(1, 2, 3)
       tester:assert(tensors[i]:getDevice() == i, "tensor does not have the correct deviceID")
    end
end

function test.multi_gpu_copy_noncontig()
   local srcDevice = 1
   local dstDevice = cutorch.getDeviceCount()

   local t1, t2
   for transposeSrc = 0,1 do
     for transposeDst = 0,1 do
        cutorch.withDevice(srcDevice,
           function() t1 = torch.CudaTensor(100000, 1000):fill(1) end)
        cutorch.withDevice(dstDevice,
           function() t2 = torch.CudaTensor(100000, 1000):fill(2) end)

        if transposeSrc == 1 then -- maybe make t1 non-contiguous
           cutorch.withDevice(srcDevice, function() t1=t1:transpose(1,2) end)
        end
        if transposeDst == 1 then -- maybe make t2 non-contiguous
           cutorch.withDevice(dstDevice, function() t2=t2:transpose(1,2) end)
        end
        cutorch.synchronize()

        -- try to induce a race on t2
        cutorch.withDevice(dstDevice, function() t2:fill(3) end)

        -- perform the copy
        -- CudaTensor:copy() should not depend on the current device
        t2:copy(t1)

        -- try to induce a race on t1
        cutorch.withDevice(srcDevice, function() t1:fill(4) end)

        -- only synchronize with dstDevice because
        -- previous line guarantees synchronization with srcDevice
        cutorch.withDevice(dstDevice, function() cutorch.synchronize() end)

        local t2_max = t2:max()
        tester:assert(t2_max == 1, "bad copy, transposeSrc= " .. transposeSrc ..
               " transposeDst= " .. transposeDst .. ". t2:max() = " .. t2_max)
      end
   end
end

function test.reset_device()
   local sz = math.floor(torch.uniform(minsize,maxsize))

   cutorch.manualSeed(2384)
   local t = torch.CudaTensor(sz):normal()

   -- Create a CPU copy and destroy the GPU tensor because the GPU pointer will be invalidated by the reset.
   local tf = t:float()
   t = nil
   collectgarbage()

   -- After a device reset, the RNG state should have been reset to its initial state.
   cutorch.deviceReset()
   local u = torch.CudaTensor(sz):normal()

   tester:assertTensorEq(tf, u:float(), 1e-6, "values not equal after restoring the RNG state")
end

function test.maskedSelect()
   local n_row = math.random(minsize,maxsize)
   local n_col = math.random(minsize,maxsize)

   -- contiguous, no result tensor, cuda mask
   local x = torch.randn(n_row, n_col):float()
   local mask = torch.ByteTensor(n_row,n_col):bernoulli()
   local y = x:maskedSelect(mask)
   x=x:cuda()
   mask=mask:cuda()
   local y_cuda = x:maskedSelect(mask)
   tester:assertTensorEq(y, y_cuda:float(), 0.00001, "Error in maskedSelect")

   -- non-contiguous, no result tensor, cuda mask
   local x = torch.randn(n_row, n_col):float()
   local mask = torch.ByteTensor(n_row,n_col):bernoulli()
   local y = x:t():maskedSelect(mask)
   x=x:cuda()
   mask=mask:cuda()
   local y_cuda = x:t():maskedSelect(mask)
   tester:assertTensorEq(y, y_cuda:float(), 0.00001, "Error in maskedSelect non-contiguous")

   -- contiguous, with result tensor, cuda mask
   local x = torch.randn(n_row, n_col):float()
   local mask = torch.ByteTensor(n_row,n_col):bernoulli()
   local y = torch.FloatTensor()
   y:maskedSelect(x, mask)
   x=x:cuda()
   mask=mask:cuda()
   local y_cuda = torch.CudaTensor()
   y_cuda:maskedSelect(x, mask)
   tester:assertTensorEq(y, y_cuda:float(), 0.00001, "Error in maskedSelect (with result)")

   -- non-contiguous, with result tensor, cuda mask
   local x = torch.randn(n_row, n_col):float()
   local mask = torch.ByteTensor(n_row,n_col):bernoulli()
   local y = torch.FloatTensor()
   y:maskedSelect(x:t(), mask)
   x=x:cuda()
   mask=mask:cuda()
   local y_cuda = torch.CudaTensor()
   y_cuda:maskedSelect(x:t(), mask)
   tester:assertTensorEq(y, y_cuda:float(), 0.00001,
			 "Error in maskedSelect non-contiguous (with result)")

   -- indexing maskedSelect a[a:gt(0.5)] for example
   local x = torch.randn(n_row, n_col):float()
   local y = x[x:gt(0.5)]
   x=x:cuda()
   mask=mask:cuda()
   local y_cuda = x[x:gt(0.5)]
   tester:assertTensorEq(y, y_cuda:float(), 0.00001, "Error in maskedSelect indexing x[x:gt(y)]")

   -- indexing maskedSelect (non-contiguous) a[a:gt(0.5)] for example
   local x = torch.randn(n_row, n_col):float()
   local y = x:t()[x:t():gt(0.5)]
   x=x:cuda()
   mask=mask:cuda()
   local y_cuda = x:t()[x:t():gt(0.5)]
   tester:assertTensorEq(y, y_cuda:float(), 0.00001,
			 "Error in maskedSelect indexing (non-contiguous) x[x:gt(y)]")
end

--[[
waiting on clarification for: https://github.com/torch/torch7/pull/187
function test.maskedCopy()
   local n_row = math.random(minsize,maxsize)
   local n_col = math.random(minsize,maxsize)

   -- contiguous, cuda mask
   local x = torch.randn(n_row, n_col):float()
   local y = x:clone():fill(-1)
   local mask = torch.ByteTensor(n_row,n_col):bernoulli()
   y:maskedCopy(mask, x:clone())
   local y_cuda=x:cuda():fill(-1)
   mask=mask:cuda()
   x=x:cuda()
   y_cuda:maskedCopy(mask, x)
   tester:assertTensorEq(y, y_cuda:float(), 0.00001, "Error in maskedCopy (contiguous)")
   -- non-contiguous source, cuda mask
   local x = torch.randn(n_row, n_col):float()
   local y = x:clone():fill(-1)
   local mask = torch.ByteTensor(n_row,n_col):bernoulli()
   y:maskedCopy(mask, x:t())
   local y_cuda=x:cuda():fill(-1)
   x=x:cuda()
   mask=mask:cuda()
   y_cuda:maskedCopy(mask, x:t())
   tester:assertTensorEq(y, y_cuda:float(), 0.00001, "Error in maskedCopy (non-contiguous source)")

   -- non-contiguous result, cuda mask
   local x = torch.randn(n_row, n_col):float()
   local y = x:clone():fill(-1)
   local mask = torch.ByteTensor(n_row,n_col):bernoulli()
   y:t():maskedCopy(mask, x:t())
   local y_cuda=x:cuda():fill(-1)
   x=x:cuda()
   mask=mask:cuda()
   y_cuda:t():maskedCopy(mask, x:t())
   tester:assertTensorEq(y, y_cuda:float(), 0.00001, "Error in maskedCopy (non-contiguous dest)")

   -- indexing maskedCopy a[a:gt(0.5)] for example
   local gt = torch.randn(n_row, n_col):float()
   local x = gt:clone()
   local y = torch.randn(n_row, n_col):float()
   x[x:gt(0.5)] = y
   local x_cuda = gt:cuda()
   y=y:cuda()
   x_cuda[x_cuda:gt(0.5)] = y
   tester:assertTensorEq(x, x_cuda:float(), 0.00001, "Error in maskedCopy indexing x[x:gt(y)]")

   -- indexing maskedCopy non-contiguous src a[a:gt(0.5)] for example
   local gt = torch.randn(n_row, n_col):float()
   local x = gt:clone()
   local y = torch.randn(n_row, n_col):float()
   x[x:gt(0.5)] = y:t()
   local x_cuda = gt:cuda()
   y=y:cuda()
   x_cuda[x_cuda:gt(0.5)] = y:t()
   tester:assertTensorEq(x, x_cuda:float(), 0.00001, "Error in maskedCopy indexing x[x:gt(y)]")

   -- indexing maskedCopy non-contiguous dst a[a:gt(0.5)] for example
   local gt = torch.randn(n_row, n_col):float()
   local x = gt:clone()
   local y = torch.randn(n_row, n_col):float()
   x:t()[x:t():gt(0.5)] = y
   local x_cuda = gt:cuda()
   y=y:cuda()
   x_cuda:t()[x_cuda:t():gt(0.5)] = y:t()
   tester:assertTensorEq(x, x_cuda:float(), 0.00001, "Error in maskedCopy indexing x[x:gt(y)]")
end
]]--

function test.maskedFill()
   local n_row = math.random(minsize,maxsize)
   local n_col = math.random(minsize,maxsize)

   -- contiguous, no result tensor, cuda mask
   local gt = torch.randn(n_row, n_col):float()
   local x = gt:clone()
   local mask = torch.ByteTensor(n_row,n_col):bernoulli()
   x:maskedFill(mask, 334)
   local x_cuda=gt:cuda()
   mask=mask:cuda()
   x_cuda:maskedFill(mask, 334)
   tester:assertTensorEq(x, x_cuda:float(), 0.00001, "Error in maskedFill")

   -- non-contiguous, no result tensor, cuda mask
   local x = gt:clone()
   mask = mask:byte()
   x:t():maskedFill(mask, 334)
   local x_cuda = gt:cuda()
   mask=mask:cuda()
   x_cuda:t():maskedFill(mask, 334)
   tester:assertTensorEq(x, x_cuda:float(), 0.00001, "Error in maskedFill non-contiguous")

   -- indexing maskedFill a[a:gt(0.5)] for example
   local x = gt:clone()
   x[x:gt(0.5)] = 334
   local x_cuda = gt:cuda()
   x_cuda[x_cuda:gt(0.5)] = 334
   tester:assertTensorEq(x, x_cuda:float(), 0.00001, "Error in maskedFill indexing x[x:gt(y)]")

   -- indexing maskedFill a[a:gt(0.5)] for example
   local x = gt:clone()
   x:t()[x:t():gt(0.5)] = 334
   local x_cuda = gt:cuda()
   x_cuda:t()[x_cuda:t():gt(0.5)] = 334
   tester:assertTensorEq(x, x_cuda:float(), 0.00001,
			 "Error in maskedFill non-contiguous indexing x[x:gt(y)]")

end

function test.sort()
   -- also tested this function with (100, 1000), but they are not
   -- good as reasonable defaults (lots of time, lots of memory)
   local minsize = 10
   local maxsize = 50
   local n_row = math.random(minsize,maxsize)
   local n_col = math.random(minsize,maxsize)
   local n_vol = math.random(minsize,maxsize)
   -- use randperm so as to not matter if stable sort or not
   local x = torch.zeros(n_vol, n_row, n_col)
   for i=1,n_vol do
         x[i]:copy(torch.randperm(n_row*n_col))
   end
   x=x:float()
   compareFloatAndCuda(x, 'sort', 3, true)
   compareFloatAndCuda(x, 'sort', 3, false)
   compareFloatAndCuda(x:transpose(2,3), 'sort', 2, true)  -- non-contiguous
   compareFloatAndCuda(x:transpose(1,3), 'sort', 1, false) -- non-contiguous

   -- in sorting dimension K, make sure you do not
   -- have equal numbers that might lead to contention (because not stable sort)
   local x2=x:transpose(2,3):clone()
   compareFloatAndCuda(x2, 'sort', 2, true)
   compareFloatAndCuda(x2, 'sort', 2, false)
   compareFloatAndCuda(x2:transpose(2,3), 'sort', 3, true)  -- non-contiguous
   compareFloatAndCuda(x2:transpose(1,3), 'sort', 2, false) -- non-contiguous


   local x1=x:transpose(1,3):clone()
   compareFloatAndCuda(x1, 'sort', 1, true)
   compareFloatAndCuda(x1, 'sort', 1, false)
   compareFloatAndCuda(x1:transpose(2,3), 'sort', 1, true)  -- non-contiguous
   compareFloatAndCuda(x1:transpose(1,3), 'sort', 3, false) -- non-contiguous
end

function cutorch.test(tests)
   math.randomseed(os.time())
   torch.manualSeed(os.time())
   cutorch.manualSeedAll(os.time())
   tester = torch.Tester()
   tester:add(test)
   tester:run(tests)
   -- print ''
   -- for module,tm in pairs(times) do
   -- print(module .. ': \t average speedup is ' .. (tm.cpu / (tm.gpu or 1e6)))
   -- end
end

if runtests then
   cutorch.test()
end
return test
