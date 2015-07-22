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
local unpack = unpack or table.unpack
--e.g. unit test cmd: th -lcutorch -e "cutorch.test{'view','viewAs'}"

-- Picks an integer between a and b, inclusive of endpoints
local function chooseInt(a, b)
   return math.floor(torch.uniform(a, b + 1))
end

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

local function checkMultiDevice(x, fn, ...)
   local device_count = cutorch.getDeviceCount()
   if device_count >= 2 then
      local x = x:cuda()
      cutorch.setDevice(cutorch.getDevice() == 1 and 2 or 1)
      local ok, err = pcall(function(...) x[fn](x, ...) end, ...)
      tester:assert(not ok, "Multi-device checks failed for: " .. tostring(fn))
      -- tester:assert(err:find("checkGPU"), "Multi-device check error message wrong for " .. tostring(fn) .. ". error: " .. tostring(err))
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
   local sz = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz, 1, sz, 1)
   compareFloatAndCuda(x, 'squeeze')

   local y = x:cuda():squeeze()
   tester:assert(y:dim() == 2, "squeeze err")

   x = torch.FloatTensor():rand(sz, 1, 1, sz)
   compareFloatAndCuda(x, 'squeeze', 2)

   local y = x:cuda():squeeze(2)
   tester:assert(y:dim() == 3, "squeeze1d err")

   x = torch.FloatTensor(1)
   compareFloatAndCuda(x, 'squeeze')
end

function test.expand()
   local sz = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz, 1)
   compareFloatAndCuda(x, 'expand', sz, sz)

   x = torch.FloatTensor():rand(1, sz)
   compareFloatAndCuda(x, 'expand', sz, sz)
end

function test.view()
   local sz = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz, 3)
   compareFloatAndCuda(x, 'view', sz, 3, 1)
end

function test.viewAs()
   local sz = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz, 3)
   local y = torch.FloatTensor():rand(sz, 3, 1)
   compareFloatAndCudaTensorArgs(x, 'viewAs', y)
end

function test.repeatTensor()
   local sz = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz, 3)
   compareFloatAndCuda(x, 'repeatTensor', sz, 2)
end

function test.permute()
   local perm = torch.randperm(7):totable()
   local x = torch.FloatTensor():rand(1, 2, 3, 4, 5, 6, 7)
   compareFloatAndCuda(x, 'permute', unpack(perm))
end

function test.split()
   local sz = {chooseInt(minsize, maxsize),
               chooseInt(minsize, maxsize),
               chooseInt(minsize, maxsize)}
   local x = torch.rand(unpack(sz))
   local dim = torch.random(3)
   local size = torch.random(sz[dim])
   local y = x:split(size, dim)
   local y_ref = x:float():split(size, dim)

   tester:asserteq(#y, #y_ref)
   for i = 1, math.min(#y, #y_ref) do
      tester:assertTensorEq(y[i]:float(), y_ref[i], 0)
   end
end

function test.chunk()
   local sz = {chooseInt(minsize, maxsize),
               chooseInt(minsize, maxsize),
               chooseInt(minsize, maxsize)}
   local x = torch.rand(unpack(sz))
   local dim = torch.random(3)
   local n = torch.random(sz[dim])
   local y = x:chunk(n, dim)
   local y_ref = x:float():chunk(n, dim)

   tester:asserteq(#y, #y_ref)
   for i = 1, math.min(#y, #y_ref) do
      tester:assertTensorEq(y[i]:float(), y_ref[i], 0)
   end
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

   local sz = chooseInt(minsize, maxsize)
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
   local sz = chooseInt(maxsize, 2 * maxsize)
   local f = function(src)
      return src.new(20, sz, 60, 60):copy(src:expand(20, sz, 60, 60))
   end
   compareFloatAndCuda(x, f)
end

function test.zero()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'zero')
   checkMultiDevice(x, 'zero')
end

function test.fill()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local v = torch.uniform()
   compareFloatAndCudaTensorArgs(x, 'fill', v)
   checkMultiDevice(x, 'fill', v)
end

function test.reshape()
   local sz1 = chooseInt(minsize, maxsize)*2
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'reshape', sz1/2, sz2*2)
   checkMultiDevice(x, 'reshape', sz1/2, sz2*2)
end

function test.zeros()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local t = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.CudaTensor')
   local x = torch.zeros(sz1, sz2)
   assert(x:sum() == 0)
   torch.setdefaulttensortype(t)
end

function test.ones()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local t = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.CudaTensor')
   local x = torch.ones(sz1, sz2)
   assert(x:sum() == x:nElement())
   torch.setdefaulttensortype(t)
end


function test.add()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor():rand(sz1, sz2)
   local v = torch.uniform()
   compareFloatAndCudaTensorArgs(x, 'add', z)
   compareFloatAndCudaTensorArgs(x, 'add', z, v)
   compareFloatAndCudaTensorArgs(x, 'add', y, z)
   compareFloatAndCudaTensorArgs(x, 'add', y, v, z)
   checkMultiDevice(x, 'add', z)
   checkMultiDevice(x, 'add', z, v)
   checkMultiDevice(x, 'add', y, z)
   checkMultiDevice(x, 'add', y, v, z)
end

function test.cmul()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'cmul', y)
   checkMultiDevice(x, 'cmul', y)
end

function test.cpow()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'cpow', y)
   checkMultiDevice(x, 'cpow', y)
end

function test.cdiv()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'cdiv', y)
   checkMultiDevice(x, 'cdiv', y)
end

function test.cdiv3()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor(sz1, sz2)
   compareFloatAndCudaTensorArgs(z, 'cdiv', x, y)
   checkMultiDevice(z, 'cdiv', x, y)
end

function test.addcmul()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'addcmul', y, z)
   compareFloatAndCudaTensorArgs(x, 'addcmul', torch.uniform(), y, z)
   checkMultiDevice(x, 'addcmul', y, z)
   checkMultiDevice(x, 'addcmul', torch.uniform(), y, z)

   local r = torch.zeros(sz1, sz2)
   compareFloatAndCudaTensorArgs(r, 'addcmul', x, y, z)
   compareFloatAndCudaTensorArgs(r, 'addcmul', x, torch.uniform(), y, z)
   checkMultiDevice(r, 'addcmul', x, y, z)
   checkMultiDevice(r, 'addcmul', x, torch.uniform(), y, z)

end

function test.addcdiv()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'addcdiv', y, z)
   compareFloatAndCudaTensorArgs(x, 'addcdiv', torch.uniform(), y, z)
   checkMultiDevice(x, 'addcdiv', y, z)
   checkMultiDevice(x, 'addcdiv', torch.uniform(), y, z)

   local r = torch.zeros(sz1, sz2)
   compareFloatAndCudaTensorArgs(r, 'addcdiv', x, y, z)
   compareFloatAndCudaTensorArgs(r, 'addcdiv', x, torch.uniform(), y, z)
   checkMultiDevice(r, 'addcdiv', x, y, z)
   checkMultiDevice(r, 'addcdiv', x, torch.uniform(), y, z)
end

function test.logicalValue()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'gt', y, 0.3)
   compareFloatAndCuda(x, 'gt', 0.3)
   checkMultiDevice(x, 'gt', y, 0.3)
   checkMultiDevice(x, 'gt', 0.3)
end

function test.logicalTensor()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCudaTensorArgs(x, 'gt', z)
   compareFloatAndCudaTensorArgs(x, 'gt', y, z)
   checkMultiDevice(x, 'gt', z)
   checkMultiDevice(x, 'gt', y, z)
end

function test.mean()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'mean')
   compareFloatAndCuda(x, 'mean', 1)
   compareFloatAndCuda(x, 'mean', 2)
   checkMultiDevice(x, 'mean')
   checkMultiDevice(x, 'mean', 1)
end

function test.max()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.randperm(sz1 * sz2):view(sz1, sz2):float()
   compareFloatAndCuda(x, 'max')
   compareFloatAndCuda(x, 'max', 1)
   compareFloatAndCuda(x, 'max', 2)
   checkMultiDevice(x, 'max')
   checkMultiDevice(x, 'max', 1)
end

function test.min()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.randperm(sz1 * sz2):view(sz1, sz2):float()
   compareFloatAndCuda(x, 'min')
   compareFloatAndCuda(x, 'min', 1)
   compareFloatAndCuda(x, 'min', 2)
   checkMultiDevice(x, 'min')
   checkMultiDevice(x, 'min', 1)
end

function test.cmax()
  local sz1 = chooseInt(minsize, maxsize)
  local sz2 = chooseInt(minsize, maxsize)
  local a = torch.FloatTensor(sz1, sz2):uniform()
  local b = torch.FloatTensor(sz1, sz2):uniform()
  local c = torch.FloatTensor(sz1, sz2):zero()
  local v = torch.uniform()

  compareFloatAndCudaTensorArgs(c, 'cmax', a, b)
  compareFloatAndCudaTensorArgs(c, 'cmax', a, v)
  compareFloatAndCudaTensorArgs(a, 'cmax', b)
  compareFloatAndCuda(a, 'cmax', v)
  checkMultiDevice(c, 'cmax', a, b)
  checkMultiDevice(c, 'cmax', a, v)
  checkMultiDevice(a, 'cmax', b)
  checkMultiDevice(a, 'cmax', v)
end

function test.cmin()
  local sz1 = chooseInt(minsize, maxsize)
  local sz2 = chooseInt(minsize, maxsize)
  local a = torch.FloatTensor(sz1, sz2):uniform()
  local b = torch.FloatTensor(sz1, sz2):uniform()
  local c = torch.FloatTensor(sz1, sz2):zero()
  local v = torch.uniform()

  compareFloatAndCudaTensorArgs(c, 'cmin', a, b)
  compareFloatAndCudaTensorArgs(c, 'cmin', a, v)
  compareFloatAndCudaTensorArgs(a, 'cmin', b)
  compareFloatAndCuda(a, 'cmin', v)
  checkMultiDevice(c, 'cmin', a, b)
  checkMultiDevice(c, 'cmin', a, v)
  checkMultiDevice(a, 'cmin', b)
  checkMultiDevice(a, 'cmin', v)
end

function test.allAndAny()
   for tries = 1, 10 do
      local size1 = chooseInt(10, 100)
      local t = nil
      if torch.uniform(0, 1) > 0.5 then
         t = torch.CudaTensor(size1):fill(1)
      else
         local size2 = chooseInt(10, 100)
         t = torch.CudaTensor(size1, size2):fill(1)

         if torch.uniform(0, 1) > 0.5 then
            t = t:transpose(1, 2)
         end
      end

      tester:assert(t:all(), 'error in all()')
      tester:assert(t:any(), 'error in any()')

      if t:dim() == 1 then
         t[chooseInt(1, t:size()[1])] = 0
      else
         t[chooseInt(1, t:size()[1])][chooseInt(1, t:size()[2])] = 0
      end

      tester:assert(not t:all(), 'error in all()')
      tester:assert(t:any(), 'error in any()')

      t:zero()
      tester:assert(not t:all(), 'error in all()')
      tester:assert(not t:any(), 'error in any()')
   end
end

function test.sum()
   local minsize = 10
   local maxsize = 20
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   test_tolerance = 1e-1
   compareFloatAndCuda(x, 'sum')
   compareFloatAndCuda(x, 'sum', 1)
   compareFloatAndCuda(x, 'sum', 2)
   test_tolerance = 1e-5
   checkMultiDevice(x, 'sum')
   checkMultiDevice(x, 'sum', 1)
end

function test.cumsum()
   local minsize = 10
   local maxsize = 20
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'cumsum')
   compareFloatAndCuda(x, 'cumsum', 1)
   compareFloatAndCuda(x, 'cumsum', 2)
   checkMultiDevice(x, 'cumsum')
   checkMultiDevice(x, 'cumsum', 1)
end

function test.prod()
   local minsize = 10
   local maxsize = 20
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'prod')
   compareFloatAndCuda(x, 'prod', 1)
   compareFloatAndCuda(x, 'prod', 2)
   checkMultiDevice(x, 'prod')
   checkMultiDevice(x, 'prod', 1)
end

function test.cumprod()
   local minsize = 10
   local maxsize = 20
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'cumprod')
   compareFloatAndCuda(x, 'cumprod', 1)
   compareFloatAndCuda(x, 'cumprod', 2)
   checkMultiDevice(x, 'cumprod')
   checkMultiDevice(x, 'cumprod', 1)
end

function test.round()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'round')
   checkMultiDevice(x, 'round')
end

function test.var()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'var')
   compareFloatAndCuda(x, 'var', 1, true)
   compareFloatAndCuda(x, 'var', 1, false)
   compareFloatAndCuda(x, 'var', 2, true)
   compareFloatAndCuda(x, 'var', 2, false)
   checkMultiDevice(x, 'var')
   checkMultiDevice(x, 'var', 1)
end

function test.std()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   compareFloatAndCuda(x, 'std')
   compareFloatAndCuda(x, 'std', 1, true)
   compareFloatAndCuda(x, 'std', 1, false)
   compareFloatAndCuda(x, 'std', 2, true)
   compareFloatAndCuda(x, 'std', 2, false)
   checkMultiDevice(x, 'std')
   checkMultiDevice(x, 'std', 1)
end

-- Test element-wise unary operators with both one and two arguments.
local function testUnary1(fn)
   local function test()
      local sz1 = chooseInt(minsize, maxsize)
      local sz2 = chooseInt(minsize, maxsize)
      local x = torch.FloatTensor():rand(sz1, sz2)
      compareFloatAndCudaTensorArgs(x, fn)
   end
   return test
end

local function testUnary2(fn)
   local function test()
      local sz1 = chooseInt(minsize, maxsize)
      local sz2 = chooseInt(minsize, maxsize)
      local x = torch.FloatTensor():rand(sz1, sz2)
      local y = torch.FloatTensor()
      compareFloatAndCudaTensorArgs(y, fn, x)
      checkMultiDevice(y, fn, x)
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
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor()
   compareFloatAndCudaTensorArgs(z, 'atan2', x, y)
   checkMultiDevice(z, 'atan2', x, y)
end

function test.pow1()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local pow = torch.uniform(minvalue,maxvalue)
   compareFloatAndCudaTensorArgs(x, 'pow', pow)
   checkMultiDevice(x, 'pow', pow)
end

function test.pow2()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor()
   local pow = torch.uniform(minvalue,maxvalue)
   compareFloatAndCudaTensorArgs(y, 'pow', x, pow)
   checkMultiDevice(y, 'pow', x, pow)
end

function test.powExponentTensor()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local pow = torch.uniform(minvalue,maxvalue)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor()
   compareFloatAndCudaTensorArgs(y, 'pow', pow, x)
   checkMultiDevice(y, 'pow', pow, x)
end

function test.clamp1()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2):mul(5):add(-2.5)
   local min_val = -1
   local max_val = 1
   x[1][1] = min_val - 1
   if sz2 >= 2 then
     x[1][2] = max_val + 1
   end
   compareFloatAndCudaTensorArgs(x, 'clamp', min_val, max_val)
   checkMultiDevice(x, 'clamp', min_val, max_val)
end

function test.clamp2()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2):mul(5):add(-2.5)
   local min_val = -1
   local max_val = 1
   x[1][1] = min_val - 1
   if sz2 >= 2 then
     x[1][2] = max_val + 1
   end
   local y = torch.FloatTensor():resizeAs(x)
   compareFloatAndCudaTensorArgs(y, 'clamp', x, min_val, max_val)
   checkMultiDevice(y, 'clamp', x, min_val, max_val)
end

function test.index()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local sz3 = chooseInt(10, 20)
   local x = torch.FloatTensor():rand(sz1, sz2)

   local longIndex = torch.LongTensor{chooseInt(1, sz1), chooseInt(1, sz1)}
   local index = 1
   compareFloatAndCuda(x, 'index', index, longIndex)

   index = 2
   longIndex =  torch.LongTensor{chooseInt(1, sz2), chooseInt(1, sz2)}
   compareFloatAndCuda(x, 'index', index, longIndex)

   x = torch.FloatTensor():rand(sz1)
   index = 1
   longIndex = torch.LongTensor{chooseInt(1, sz1), chooseInt(1, sz1)}
   compareFloatAndCuda(x, 'index', index, longIndex)

   x = torch.FloatTensor():rand(sz1,sz2,sz3)
   index = 3
   longIndex = torch.randperm(sz3):long()
   compareFloatAndCuda(x, 'index', index, longIndex)

   tester:assert(isEqual(x:cuda():index(index, longIndex:cuda()), x:index(index, longIndex)),
      "Divergent results between CPU and CUDA for function 'index'")

   checkMultiDevice(x, 'index', index, longIndex)
end

function test.indexCopy()
   local sz1 = chooseInt(minsize, maxsize) -- dim1
   local sz2 = chooseInt(minsize, maxsize) -- dim2
   local x = torch.FloatTensor():rand(sz1, sz2) -- input


   -- Case 1: 2D tensor, indexCopy over first dimension, 2 indices
   -- choose two indices from the first dimension, i.e. [1,sz1]
   local longIndex = torch.LongTensor{chooseInt(1, sz1), chooseInt(1, sz1)}
   local index = 1
   local src = torch.FloatTensor(2, sz2):uniform()
   compareFloatAndCudaTensorArgs(x, 'indexCopy', index, longIndex, src)

   -- Case 2: 2D tensor, indexCopy over second dimension, 2 indices
   index = 2
   longIndex =  torch.LongTensor{chooseInt(1, sz2), chooseInt(1, sz2)}
   src = torch.FloatTensor(sz1, 2):uniform():cuda()
   compareFloatAndCudaTensorArgs(x, 'indexCopy', index, longIndex, src)

   -- Case 3: 1D tensor, indexCopy over 1st dimension, 2 indices
   x = torch.FloatTensor():rand(sz1)
   index = 1
   longIndex = torch.LongTensor{chooseInt(1, sz1), chooseInt(1, sz1)}
   src = torch.FloatTensor(2):uniform()
   compareFloatAndCudaTensorArgs(x, 'indexCopy', index, longIndex, src)

   tester:assert(isEqual(
      x:cuda():indexCopy(index, longIndex:cuda(), src:cuda()),
      x:indexCopy(index, longIndex, src)),
      "Divergent results between CPU and CUDA for function 'indexCopy'")

   checkMultiDevice(x, 'indexCopy', index, longIndex, src)
end

function test.indexFill()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)

   local longIndex = torch.LongTensor{chooseInt(1, sz1), chooseInt(1, sz1)}
   local index = 1
   local val = torch.randn(1)[1]
   compareFloatAndCuda(x, 'indexFill', index, longIndex, val)

   index = 2
   longIndex =  torch.LongTensor{chooseInt(1, sz2), chooseInt(1, sz2)}
   val = torch.randn(1)[1]
   compareFloatAndCuda(x, 'indexFill', index, longIndex, val)

   x = torch.FloatTensor():rand(sz1)
   index = 1
   longIndex = torch.LongTensor{chooseInt(1, sz1), chooseInt(1, sz1)}
   val = torch.randn(1)[1]
   compareFloatAndCuda(x, 'indexFill', index, longIndex, val)

   tester:assert(isEqual(
      x:cuda():indexFill(index, longIndex:cuda(), val),
      x:indexFill(index, longIndex, val)),
      "Divergent results between CPU and CUDA for function 'indexFill'")

   checkMultiDevice(x, 'indexFill', index, longIndex, val)
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

   checkMultiDevice(x, 'renorm', 4, 2, maxnorm)
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
   local multiCheck = false
   for _, size in pairs(sizes) do
      local n, m = unpack(size)
      local c = torch.zeros(n)
      local a = torch.randn(n, m)
      local b = torch.randn(m)
      compareFloatAndCudaTensorArgs(c, 'addmv', torch.normal(), torch.normal(), a, b)
      if not multiCheck then -- just check multidevice once
         checkMultiDevice(c, 'addmv', torch.normal(), torch.normal(), a, b)
         multiCheck = true
      end
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
   local multiCheck = false
   for _, size in pairs(sizes) do
      local n, m = unpack(size)
      local c = torch.zeros(n)
      local a = torch.randn(n, m)
      local b = torch.randn(m)
      compareFloatAndCudaTensorArgs(c, 'mv', a, b)
      if not multiCheck then -- just check multidevice once
         checkMultiDevice(c, 'mv', a, b)
         multiCheck = true
      end
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
   local multiCheck = false
   for _, size in pairs(sizes) do
      local n, m = unpack(size)
      local c = torch.zeros(n,m)
      local a = torch.randn(n)
      local b = torch.randn(m)
      compareFloatAndCudaTensorArgs(c, 'addr', torch.normal(), a, b)
      if not multiCheck then -- just check multidevice once
         checkMultiDevice(c, 'addr', torch.normal(), a, b)
         multiCheck = true
      end
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
   local multiCheck = false
   for _, size in pairs(sizes) do
      local n, k, m = unpack(size)
      local c = torch.zeros(n, m)
      local a = torch.randn(n, k)
      local b = torch.randn(k, m)
      compareFloatAndCudaTensorArgs(c, 'addmm', torch.normal(), torch.normal(), a, b)
      if not multiCheck then -- just check multidevice once
         checkMultiDevice(c, 'addmm', torch.normal(), torch.normal(), a, b)
         multiCheck = true
      end
   end

   -- check all zero-strided cases for the inputs
   -- considers that the output tensor is not zero-strided
   local n, k, m = 10, 10, 10
   local function generateTensor(t,idx)
      local tensor = torch.FloatTensor()
      local s1,s2
      if t == 1 then
        s1 = n
        s2 = m
      elseif t == 2 then
        s1 = n
        s2 = k
      else
        s1 = k
        s2 = m
      end
      if idx == 1 then
        tensor:resize(s1,s2)
      elseif idx == 2 then
        tensor:resize(s1,1)
      elseif idx == 3 then
        tensor:resize(1,s2)
      else
        tensor:resize(1,1)
      end
      if t == 1 then
        tensor:zero()
      else
        tensor:uniform()
      end
      tensor = tensor:expand(s1,s2)
      return tensor
   end
   
   for i = 1, 4*4*4 do
      local a_idx = (i-1)%4 + 1
      local b_idx = math.floor(((i-1)%16)/4)  + 1
      local c_idx = 1 -- math.floor((i-1)/16) + 1
      local c = generateTensor(1,c_idx)
      local a = generateTensor(2,a_idx)
      local b = generateTensor(3,b_idx)
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
   local multiCheck = false
   for _, size in pairs(sizes) do
      local n, k, m = unpack(size)
      local c = torch.zeros(n, m)
      local a = torch.randn(n, k)
      local b = torch.randn(k, m)
      compareFloatAndCudaTensorArgs(c, 'mm', a, b)
      if not multiCheck then -- just check multidevice once
         checkMultiDevice(c, 'mm', a, b)
         multiCheck = true
      end
   end

   -- check all zero-strided cases for the inputs
   -- considers that the output tensor is not zero-strided
   local n, k, m = 10, 10, 10
   local function generateTensor(t,idx)
      local tensor = torch.FloatTensor()
      local s1,s2
      if t == 1 then
        s1 = n
        s2 = m
      elseif t == 2 then
        s1 = n
        s2 = k
      else
        s1 = k
        s2 = m
      end
      if idx == 1 then
        tensor:resize(s1,s2)
      elseif idx == 2 then
        tensor:resize(s1,1)
      elseif idx == 3 then
        tensor:resize(1,s2)
      else
        tensor:resize(1,1)
      end
      if t == 1 then
        tensor:zero()
      else
        tensor:uniform()
      end
      tensor = tensor:expand(s1,s2)
      return tensor
   end
   
   for i = 1, 4*4*4 do
      local a_idx = (i-1)%4 + 1
      local b_idx = math.floor(((i-1)%16)/4)  + 1
      local c_idx = 1 -- math.floor((i-1)/16) + 1
      local c = generateTensor(1,c_idx)
      local a = generateTensor(2,a_idx)
      local b = generateTensor(3,b_idx)
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
   local multiCheck = false
   for _, size in pairs(sizes) do
      local b, n, k, m = unpack(size)
      local cs = torch.randn(b, n, m)
      local as = torch.randn(b, n, k)
      local bs = torch.randn(b, k, m)
      compareFloatAndCudaTensorArgs(cs, 'baddbmm', as, bs)
      if not multiCheck then -- just check multidevice once
         checkMultiDevice(cs, 'baddbmm', as, bs)
         multiCheck = true
      end
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
   local multiCheck = false
   for _, size in pairs(sizes) do
      local b, n, k, m = unpack(size)
      local cs = torch.zeros(b, n, m)
      local as = torch.randn(b, n, k)
      local bs = torch.randn(b, k, m)
      compareFloatAndCudaTensorArgs(cs, 'bmm', as, bs)
      if not multiCheck then -- just check multidevice once
         checkMultiDevice(cs, 'bmm', as, bs)
         multiCheck = true
      end
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
   local multiCheck = false
   for _, size in pairs(sizes) do
      local n, m = unpack(size)
      local c = torch.zeros(n, m)
      local a = torch.randn(n)
      local b = torch.randn(m)
      compareFloatAndCudaTensorArgs(c, 'ger', a, b)
      if not multiCheck then -- just check multidevice once
         checkMultiDevice(c, 'ger', a, b)
         multiCheck = true
      end
   end
end

if cutorch.magma then
   function test.gesv()
      local a = torch.Tensor(5, 5):uniform(-1, 1)
      local b = torch.Tensor(5, 3):uniform(-1, 1)
      local rb1, ra1 = torch.gesv(b, a)
      local rb2, ra2 = torch.gesv(b:cuda(), a:cuda())
      tester:assertle((rb2 - rb1:cuda()):abs():max(), 1e-5, "wrong gesv answer")
      tester:assertle((ra2 - ra1:cuda()):abs():max(), 1e-5, "wrong gesv answer")
   end

   function test.gels()
      local a = torch.Tensor{
         {-0.8862, 0.8186,  0.2334,  0.8008,  0.2377},
         { 0.6116, 0.2242,  0.2854,  0.5427,  0.5937},
         {-0.3716,-0.7247, -0.7658, -0.1285,  0.6749},
         {-0.5878, 0.7596, -0.7765, -0.5373,  0.6326},
         { 0.0868,-0.4918,  0.7771, -0.7550, -0.6020},
      }
      local b = torch.Tensor{
         { 0.4807, 0.1842, 0.7908},
         {-0.0035, 0.7557, 0.1627},
         { 0.3495,-0.0840, 0.8164},
         { 0.5360, 0.2048, 0.2745},
         { 0.8535,-0.3938,-0.2140},
      }
      local rb1, ra1 = torch.gels(b, a)
      local rb2, ra2 = torch.gels(b:cuda(), a:cuda())
      tester:assertle((rb2 - rb1:cuda()):abs():max(), 5e-4, "wrong gels answer")
      tester:assertle((ra2 - ra1:cuda()):abs():max(), 5e-4, "wrong gels answer")
   end

   function test.symeig()
      local a = torch.Tensor({{ 1.96,  0.00,  0.00,  0.00,  0.00},
                              {-6.49,  3.80,  0.00,  0.00,  0.00},
                              {-0.47, -6.39,  4.17,  0.00,  0.00},
                              {-7.20,  1.50, -1.51,  5.70,  0.00},
                              {-0.65, -6.34,  2.67,  1.80, -7.10}}):t()
      local e1,v1 = torch.symeig(a, 'V')
      local e2,v2 = torch.symeig(a:cuda(), 'V')
      tester:assertle((e2 - e1:cuda()):abs():max(), 1e-5, "wrong symeig answer")
      tester:assertle((v2 - v1:cuda()):abs():max(), 1e-5, "wrong symeig answer")
   end

   function test.eig()
      local a = torch.Tensor{
         {-0.1425, -0.4750, -0.8551, 0.6729, -0.7453},
         {-0.2696,  0.4330,  0.5077, 0.3709, -0.6053},
         { 0.4330,  0.6727, -0.5049, 0.4600,  0.6249},
         { 0.5766, -0.6743,  0.6903, 0.3646, -0.4571},
         {-0.8956, -0.4074, -0.7583, 0.1838, -0.0091},
      }
      local e1,v1 = torch.eig(a, 'V')
      local e2,v2 = torch.eig(a:cuda(), 'V')
      tester:assertle((e2 - e1:cuda()):abs():max(), 1e-6, "wrong eig answer")
      tester:assertle((v2:abs() - v1:abs():cuda()):abs():max(), 1e-6, "wrong eig answer")
   end

   function test.svd()
      local a = torch.CudaTensor{
         {8.79,  6.11, -9.15,  9.57, -3.49,  9.84},
         {9.93,  6.91, -7.93,  1.64,  4.02,  0.15},
         {9.83,  5.04,  4.86,  8.83,  9.80, -8.99},
         {5.45, -0.27,  4.85,  0.74, 10.00, -6.02},
         {3.16,  7.98,  3.01,  5.80,  4.27, -5.31}}

      local u,s,v = torch.svd(a, 'A')

      local temp = torch.Tensor(a:size(2)):zero()
      temp:narrow(1, 1, a:size(1)):copy(s)
      local sigma = torch.diag(temp):resize(a:size(1), a:size(2)):cuda()

      local m = u * sigma * v:t()

      tester:assertle((m - a):abs():max(), 1e-5, "svd: a != u * s * vT")
      tester:assertle((u*u:t() - torch.eye(a:size(1)):cuda()):abs():max(), 1e-6, "svd: u should be unitary")
      tester:assertle((v*v:t() - torch.eye(a:size(2)):cuda()):abs():max(), 1e-6, "svd: v should be unitary")
   end

   function test.inverse()
      local a = torch.randn(5, 5)
      local i1 = torch.inverse(a)
      local i2 = torch.inverse(a:cuda())
      tester:assertle((i2 - i1:cuda()):abs():max(), 1e-5, "wrong inverse answer")
   end

   function test.potri()
      local A = torch.Tensor{
         { 0.9023,  1.5967,  0.3388, -0.0746, -0.5717},
         {-2.0442,  2.3974, -1.0883,  0.4018, -0.3938},
         {-0.1065, -1.3180,  0.3542,  1.3684,  0.3934},
         {-0.2987,  1.9035, -1.4192, -0.9738,  1.4384},
         {-0.5315,  0.4958,  0.4449, -0.4676, -0.4878},
      }
      A = A * A:t()

      local i1 = torch.potri(A)
      local i2 = torch.potri(A:cuda())
      local M = A:cuda() * i2
      tester:assertle((i2 - i1:cuda()):abs():max(), 1e-5, "wrong potri answer")
      tester:assertle((M - torch.eye(A:size(1)):cuda()):abs():max(), 1e-5, "potri not an inverse")
   end

   function test.potrf()
      local A = torch.Tensor{
         { 8.7937, 0.5104, 1.5955,-0.6738,-3.3883},
         { 0.5104, 1.4286, 0.0236, 0.4734, 0.2807},
         { 1.5955, 0.0236, 1.4539,-1.1123, 0.8161},
         {-0.6738, 0.4734,-1.1123, 2.4071,-1.2756},
         {-3.3883, 0.2807, 0.8161,-1.2756, 4.3415},
      }
      local i1 = torch.potrf(A)
      local i2 = torch.potrf(A:cuda())
      tester:assertle((i2 - i1:cuda()):abs():max(), 1e-5, "wrong potrf answer")
   end

   function test.qr()
      local A = torch.Tensor{
         { 0.9023,  1.5967,  0.3388, -0.0746, -0.5717},
         {-2.0442,  2.3974, -1.0883,  0.4018, -0.3938},
         {-0.1065, -1.3180,  0.3542,  1.3684,  0.3934},
         {-0.2987,  1.9035, -1.4192, -0.9738,  1.4384},
         {-0.5315,  0.4958,  0.4449, -0.4676, -0.4878},
      }
      local q1,r1 = torch.qr(A)
      local q2,r2 = torch.qr(A:cuda())
      tester:assertle((q2 - q1:cuda()):abs():max(), 1e-5, "wrong qr answer")
      tester:assertle((r2 - r1:cuda()):abs():max(), 1e-5, "wrong qr answer")
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
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local min = torch.uniform()
   local max = min + torch.uniform()
   local t = torch.CudaTensor(sz1, sz2)

   t:uniform(min, max)
   checkIfUniformlyDistributed(t, min, max)
   checkMultiDevice(t, 'uniform', min, max)
end

function test.bernoulli()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local p = torch.uniform()
   local t = torch.CudaTensor(sz1, sz2)

   t:bernoulli(p)
   tester:assertalmosteq(t:mean(), p, 0.1, "mean is not equal to p")
   local f = t:float()
   tester:assertTensorEq(f:eq(1):add(f:eq(0)):float(),
                         torch.FloatTensor(sz1, sz2):fill(1),
                         1e-6,
                         "each value must be either 0 or 1")
   checkMultiDevice(t, 'bernoulli', p)
end

function test.normal()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local mean, std = torch.uniform(), 0.1 * torch.uniform()
   local tolerance = 0.01
   local t = torch.CudaTensor(sz1, sz2)

   t:normal(mean, std)
   tester:assertalmosteq(t:mean(), mean, tolerance, "mean is wrong")
   tester:assertalmosteq(t:std(), std, tolerance, "standard deviation is wrong")
   checkMultiDevice(t, 'normal', mean, std)
end

function test.logNormal()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local mean, std = torch.uniform(), 0.1 * torch.uniform()
   local tolerance = 0.01
   local t = torch.CudaTensor(sz1, sz2)

   t:logNormal(mean, std)
   local logt = t:log()
   tester:assertalmosteq(logt:mean(), mean, tolerance, "mean is wrong")
   tester:assertalmosteq(logt:std(), std, tolerance, "standard deviation is wrong")
   checkMultiDevice(t, 'logNormal', mean, std)
end

function test.geometric()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local p = torch.uniform()
   local t = torch.CudaTensor(sz1, sz2)

   t:geometric(p)
   local u = torch.FloatTensor(sz1, sz2):fill(1) -
                 ((t:float() - 1) * math.log(p)):exp()
   checkIfUniformlyDistributed(u, 0, 1)
   checkMultiDevice(t, 'geometric', p)
end

function test.exponential()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local lambda = torch.uniform()
   local t = torch.CudaTensor(sz1, sz2)

   t:exponential(lambda)
   local u = torch.FloatTensor(sz1, sz2):fill(1) -
                 (t:float() * -lambda):exp()
   checkIfUniformlyDistributed(u, 0, 1)
   checkMultiDevice(t, 'exponential', lambda)
end

function test.cauchy()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local median, sigma = torch.uniform(), torch.uniform()
   local t = torch.CudaTensor(sz1, sz2)

   t:cauchy(median, sigma)
   local u = ((t:float() - median) / sigma):atan() / math.pi + 0.5
   checkIfUniformlyDistributed(u, 0, 1)
   checkMultiDevice(t, 'cauchy', median, sigma)
end

function test.random_seed()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
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
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
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
   cutorch.setDevice(1) -- reset device
end

function test.multinomial_with_replacement()
   for tries = 1, 10 do
      local n_row = torch.random(10)
      local n_col = 1 + torch.random(1000)

      local prob_dist = torch.CudaTensor(n_row, n_col):uniform()
      prob_dist:select(2, n_col):fill(0) --index n_col shouldn't be sampled
      local n_sample = torch.random(n_col - 1)
      local sample_indices = torch.multinomial(prob_dist, n_sample, true)
      tester:assert(sample_indices:dim() == 2, "wrong sample_indices dim")
      tester:assert(sample_indices:size(2) == n_sample, "wrong number of samples")

      for i = 1, n_row do
         for j = 1, n_sample do
            local val = sample_indices[{i,j}]
            tester:assert(val == math.floor(val) and val >= 1 and val < n_col,
                          "sampled an invalid index: " .. val)
         end
      end
   end
end

function test.multinomial_without_replacement()
   for tries = 1, 10 do
      local n_row = torch.random(1000)
      -- choose a small number of columns to test that the 0 col is never chosen
      local n_col = 1 + torch.random(10)

      local prob_dist = torch.CudaTensor(n_row, n_col):uniform()
      prob_dist:select(2, n_col):fill(0) --index n_col shouldn't be sampled
      local n_sample = torch.random(n_col - 1)
      local sample_indices = torch.multinomial(prob_dist, n_sample, false)
      tester:assert(sample_indices:dim() == 2, "wrong sample_indices dim")
      tester:assert(sample_indices:size(2) == n_sample, "wrong number of samples")

      sample_indices = sample_indices:float()

      for i = 1, n_row do
         local row_samples = {}
         for j = 1, n_sample do
            local sample_idx = sample_indices[{i,j}]
            tester:assert(
               sample_idx ~= n_col, "sampled an index with zero probability"
            )
            tester:assert(
                  not row_samples[sample_idx], "sampled an index twice"
            )
            row_samples[sample_idx] = true
         end
      end
   end
end

function test.multinomial_without_replacement_gets_all()
   for tries = 1, 10 do
      local distributions = torch.random(10)
      local distSize = 1 + torch.random(1000)

      local linear = torch.linspace(1, distSize, distSize):cuda()
      local t = torch.CudaTensor(distributions, distSize)
      for dist = 1, distributions do
         t[dist] = linear
      end

      local orig = t:clone()

      -- Sample without replacement
      local result = torch.multinomial(t, distSize)
      tester:assert(result:size(1) == distributions)
      tester:assert(result:size(2) == distSize)

      -- Sort, and we should have the original results, since without replacement
      -- sampling everything, we should have chosen every value uniquely
      result = result:sort(2)
      tester:assert(orig, result, 0)
   end
end

function test.multinomial_vector()
   local n_col = torch.random(100)
   local prob_dist = torch.CudaTensor(n_col):uniform()
   local n_sample = n_col
   local sample_indices = torch.multinomial(prob_dist, n_sample, true)
   tester:assert(sample_indices:dim() == 1, "wrong sample_indices dim")
   -- Multinomial resizes prob_dist to be 2d (1xn), check that the resize
   -- was undone
   tester:assert(prob_dist:dim() == 1, "wrong number of prob_dist dimensions")
   tester:assert(sample_indices:size(1) == n_sample, "wrong number of samples")
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
    cutorch.setDevice(1) -- reset device
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

        local t2_max
        cutorch.withDevice(dstDevice, function() t2_max = t2:max() end)
        tester:assert(t2_max == 1, "bad copy, transposeSrc= " .. transposeSrc ..
               " transposeDst= " .. transposeDst .. ". t2:max() = " .. t2_max)
      end
   end
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
   checkMultiDevice(x, 'maskedSelect', mask)

   -- non-contiguous, no result tensor, cuda mask
   local x = torch.randn(n_row, n_col):float()
   local mask = torch.ByteTensor(n_row,n_col):bernoulli()
   local y = x:t():maskedSelect(mask)
   x=x:cuda()
   mask=mask:cuda()
   local y_cuda = x:t():maskedSelect(mask)
   tester:assertTensorEq(y, y_cuda:float(), 0.00001,
                         "Error in maskedSelect non-contiguous")

   -- contiguous, with result tensor, cuda mask
   local x = torch.randn(n_row, n_col):float()
   local mask = torch.ByteTensor(n_row,n_col):bernoulli()
   local y = torch.FloatTensor()
   y:maskedSelect(x, mask)
   x=x:cuda()
   mask=mask:cuda()
   local y_cuda = torch.CudaTensor()
   y_cuda:maskedSelect(x, mask)
   tester:assertTensorEq(y, y_cuda:float(), 0.00001,
                         "Error in maskedSelect (with result)")

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
   local y_cuda = x[x:gt(0.5)]
   tester:assertTensorEq(y, y_cuda:float(), 0.00001,
                         "Error in maskedSelect indexing x[x:gt(y)]")

   -- indexing maskedSelect (non-contiguous) a[a:gt(0.5)] for example
   local x = torch.randn(n_row, n_col):float()
   local y = x:t()[x:t():gt(0.5)]
   x=x:cuda()
   local y_cuda = x:t()[x:t():gt(0.5)]
   tester:assertTensorEq(y, y_cuda:float(), 0.00001,
          "Error in maskedSelect indexing non-contig x[x:gt(y)]")
end

function test.maskedCopy()
   local n_row = math.random(minsize,maxsize)
   local n_col = math.random(minsize,maxsize)

   -- contiguous, cuda mask
   local x = torch.rand(n_row, n_col):float()
   local y = x:clone():fill(-1)
   local mask = torch.ByteTensor(n_row,n_col):bernoulli()
   y:maskedCopy(mask, x:clone())
   local y_cuda=x:cuda():fill(-1)
   mask=mask:cuda()
   x=x:cuda()
   y_cuda:maskedCopy(mask, x)
   tester:assertTensorEq(y, y_cuda:float(), 0.00001,
                                 "Error in maskedCopy (contiguous)")
   checkMultiDevice(y_cuda, 'maskedCopy', mask, x)

   -- non-contiguous source, cuda mask
   local x = torch.rand(n_row, n_col):float()
   local y = x:clone():fill(-1)
   local mask = torch.ByteTensor(n_row,n_col):bernoulli()
   y:maskedCopy(mask, x:t())
   local y_cuda=x:cuda():fill(-1)
   x=x:cuda()
   mask=mask:cuda()
   y_cuda:maskedCopy(mask, x:t())
   tester:assertTensorEq(y, y_cuda:float(), 0.00001,
                       "Error in maskedCopy (non-contiguous source)")

   -- non-contiguous result, cuda mask
   local x = torch.rand(n_row, n_col):float()
   local y = x:clone():fill(-1)
   local mask = torch.ByteTensor(n_row,n_col):bernoulli()
   y:t():maskedCopy(mask, x:t())
   local y_cuda=x:cuda():fill(-1)
   x=x:cuda()
   mask=mask:cuda()
   y_cuda:t():maskedCopy(mask, x:t())
   tester:assertTensorEq(y, y_cuda:float(), 0.00001,
                        "Error in maskedCopy (non-contiguous dest)")

   -- indexing maskedCopy a[a:gt(0.5)] for example
   local gt = torch.rand(n_row, n_col):float()
   local x = gt:clone()
   local y = torch.rand(n_row, n_col):float()
   x[x:gt(0.5)] = y
   local x_cuda = gt:cuda()
   y=y:cuda()
   x_cuda[x_cuda:gt(0.5)] = y
   tester:assertTensorEq(x, x_cuda:float(), 0.00001,
                             "Error in maskedCopy indexing x[x:gt(y)]")

   -- indexing maskedCopy non-contiguous src a[a:gt(0.5)] for example
   local gt = torch.rand(n_row, n_col):float()
   local x = gt:clone()
   local y = torch.rand(n_row, n_col):float()
   x[x:gt(0.5)] = y:t()
   local x_cuda = gt:cuda()
   y=y:cuda()
   x_cuda[x_cuda:gt(0.5)] = y:t()
   tester:assertTensorEq(x, x_cuda:float(), 0.00001,
                            "Error in maskedCopy indexing x[x:gt(y)]")

   -- indexing maskedCopy non-contiguous dst a[a:gt(0.5)] for example
   local gt = torch.rand(n_row, n_col):float()
   local x = gt:clone()
   local y = torch.rand(n_row, n_col):float()
   x:t()[x:t():gt(0.5)] = y
   local x_cuda = gt:cuda()
   y=y:cuda()
   x_cuda:t()[x_cuda:t():gt(0.5)] = y

   tester:assertTensorEq(x, x_cuda:float(), 0.00001,
                         "Error in maskedCopy indexing x[x:gt(y)]")
end

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
   checkMultiDevice(x_cuda, 'maskedFill', mask, 334)

   -- non-contiguous, no result tensor, cuda mask
   local x = gt:clone()
   mask = mask:byte()
   x:t():maskedFill(mask, 334)
   local x_cuda = gt:cuda()
   mask=mask:cuda()
   x_cuda:t():maskedFill(mask, 334)
   tester:assertTensorEq(x, x_cuda:float(), 0.00001,
                         "Error in maskedFill non-contiguous")

   -- indexing maskedFill a[a:gt(0.5)] for example
   local x = gt:clone()
   x[x:gt(0.5)] = 334
   local x_cuda = gt:cuda()
   x_cuda[x_cuda:gt(0.5)] = 334
   tester:assertTensorEq(x, x_cuda:float(), 0.00001,
                         "Error in maskedFill indexing x[x:gt(y)]")

   -- indexing maskedFill a[a:gt(0.5)] for example
   local x = gt:clone()
   x:t()[x:t():gt(0.5)] = 334
   local x_cuda = gt:cuda()
   x_cuda:t()[x_cuda:t():gt(0.5)] = 334
   tester:assertTensorEq(x, x_cuda:float(), 0.00001,
          "Error in maskedFill non-contig indexing x[x:gt(y)]")

end

-- Fill idx with valid indices.
local function fillIdx(idx, dim, dim_size, elems_per_row, m, n, o)
   for i = 1, (dim == 1 and 1 or m) do
      for j = 1, (dim == 2 and 1 or n) do
         for k = 1, (dim == 3 and 1 or o) do
            local ii = {i, j, k}
            ii[dim] = {}
            idx[ii] = torch.randperm(dim_size)[{{1, elems_per_row}}]
         end
      end
   end
end

function test.gather()
   local m, n, o = torch.random(10, 20), torch.random(10, 20), torch.random(10, 20)
   local elems_per_row = torch.random(10)
   local dim = torch.random(3)

   local src = torch.randn(m, n, o):float()
   local idx_size = {m, n, o}
   idx_size[dim] = elems_per_row
   local idx = torch.LongTensor():resize(unpack(idx_size))
   fillIdx(idx, dim, src:size(dim), elems_per_row, m, n, o)

   local actual = torch.gather(src:cuda(), dim, idx:cuda())
   local expected = torch.gather(src, dim, idx)
   tester:assertTensorEq(actual:float(), expected, 0, "Wrong values for gather")
end

function test.scatter()
   local m, n, o = torch.random(10, 20), torch.random(10, 20), torch.random(10, 20)
   local elems_per_row = torch.random(10)
   local dim = torch.random(3)

   local idx_size = {m, n, o}
   idx_size[dim] = elems_per_row
   local idx = torch.LongTensor():resize(unpack(idx_size))
   fillIdx(idx, dim, ({m, n, o})[dim], elems_per_row, m, n, o)
   local src = torch.FloatTensor():resize(unpack(idx_size)):normal()

   local actual = torch.CudaTensor(m, n, o):zero():scatter(dim, idx:cuda(), src:cuda())
   local expected = torch.FloatTensor(m, n, o):zero():scatter(dim, idx, src)
   tester:assertTensorEq(actual:float(), expected, 0, "Wrong values for scatter")
end

function test.scatterFill()
   local m, n, o = torch.random(10, 20), torch.random(10, 20), torch.random(10, 20)
   local elems_per_row = torch.random(10)
   local dim = torch.random(3)

   local val = torch.uniform()
   local idx_size = {m, n, o}
   idx_size[dim] = elems_per_row
   local idx = torch.LongTensor():resize(unpack(idx_size))
   fillIdx(idx, dim, ({m, n, o})[dim], elems_per_row, m, n, o)

   local actual = torch.CudaTensor(m, n, o):zero():scatter(dim, idx:cuda(), val)
   local expected = torch.FloatTensor(m, n, o):zero():scatter(dim, idx, val)
   tester:assertTensorEq(actual:float(), expected, 0, "Wrong values for scatter")
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
   checkMultiDevice(x1, 'sort', 1, true)
end

function test.streamWaitFor()
   local size = 2000000
   local iter = 20 + torch.random(10)
   local result = torch.CudaTensor(size):zero()
   local numStreams = torch.random(10)

   cutorch.reserveStreams(numStreams + 1)
   local tensors = {}
   local waitingFor = {}

   for stream = 1, numStreams do
      cutorch.setStream(stream)
      table.insert(waitingFor, stream)
      table.insert(tensors, torch.CudaTensor(size):zero())
   end

   -- Queue a bunch of work on different streams
   for stream = 1, numStreams do
      cutorch.setStream(stream)
      for i = 1, iter do
         tensors[stream]:add(1)
      end
   end

   -- In another stream, wait on the completion of all the above.
   -- Without the streamWaitFor, this will race with the above and won't
   -- gather all of the additions.
   -- Unfortunately, it would be rather hard to write a test to ensure that
   -- we're actually executing all this asynchronously, and to write a test that
   -- always guarantees failure with this race is equally problematic.
   -- So, we satisfy ourselves with this.
   cutorch.setStream(numStreams + 1)
   cutorch.streamWaitFor(numStreams + 1, waitingFor)

   for i = 1, numStreams do
      result:add(tensors[i])
   end

   tester:asserteq(result:min(), iter * numStreams)

   -- return to default stream
   cutorch.setStream(0)
   result = nil
   tensors = nil
   collectgarbage()
   collectgarbage()
   cutorch.synchronize()
end

function test.streamWaitForMultiDevice()
   -- This test requires multiple devices
   local numDevices = cutorch.getDeviceCount()
   if numDevices < 2 then
      return
   end

   local size = 2000000
   local iter = 20 + torch.random(10)
   local result = torch.CudaTensor(size):zero()
   local numStreams = torch.random(10)
   cutorch.reserveStreams(numStreams + 1)

   -- Create scratch space on the last device to receive all results
   cutorch.setDevice(numDevices)
   local tmpResults = {}
   local results = torch.CudaTensor(size):zero()

   for dev = 1, numDevices - 1 do
      local tmpResultsPerDevice = {}
      for stream = 1, numStreams do
         table.insert(tmpResultsPerDevice, torch.CudaTensor(size):zero())
      end

      table.insert(tmpResults, tmpResultsPerDevice)
   end

   -- Allocate data on all devices (except the last)
   local tensors = {}
   local waitingOn = {}

   for dev = 1, numDevices - 1 do
      cutorch.setDevice(dev)
      local tensorsPerDevice = {}
      local waitingOnPerDevice = {}

      for stream = 1, numStreams do
         cutorch.setStream(stream)
         table.insert(tensorsPerDevice, torch.CudaTensor(size):zero())
         table.insert(waitingOnPerDevice, stream)
      end

      table.insert(tensors, tensorsPerDevice)
      table.insert(waitingOn, waitingOnPerDevice)
   end

   -- Queue work to all streams, all devices (except the last)
   for dev = 1, numDevices - 1 do
      cutorch.setDevice(dev)
      for stream = 1, numStreams do
         cutorch.setStream(stream)
         for i = 1, iter do
            tensors[dev][stream]:add(1)
         end
      end
   end

   -- Cross-device/stream wait
   cutorch.streamWaitForMultiDevice(numDevices, numStreams + 1, waitingOn)

   -- Copy back to device `numDevices`, stream (numStreams + 1)
   cutorch.setDevice(numDevices)
   cutorch.setStream(numStreams + 1)

   for dev = 1, numDevices - 1 do
      for stream = 1, numStreams do
         tmpResults[dev][stream]:copy(tensors[dev][stream])
      end
   end

   -- Sum up the results
   for dev = 1, numDevices - 1 do
      for stream = 1, numStreams do
         results:add(tmpResults[dev][stream])
      end
   end

   tester:asserteq(results:min(), iter * numStreams * (numDevices - 1))

   -- return to default device/stream
   cutorch.setDevice(1)
   cutorch.setStream(0)
   result = nil
   tmpResults = nil
   tensors = nil
   collectgarbage()
   collectgarbage()
   cutorch.synchronize()
end

function test.streamBarrier()
   local size = 2000000
   local iter = 20 + torch.random(10)
   local numStreams = torch.random(10)

   cutorch.reserveStreams(numStreams)
   local tensors = {}
   local results = {}
   local waitingFor = {}

   for stream = 1, numStreams do
      cutorch.setStream(stream)
      table.insert(waitingFor, stream)
      table.insert(tensors, torch.CudaTensor(size):zero())
      table.insert(results, torch.CudaTensor(size):zero())
   end

   -- Queue a bunch of work on different streams
   for stream = 1, numStreams do
      cutorch.setStream(stream)
      for i = 1, iter do
         tensors[stream]:add(1)
      end
   end

   -- Create an all-way barrier
   cutorch.streamBarrier(waitingFor)

   -- In all streams, sum against all other tensors
   for stream = 1, numStreams do
      cutorch.setStream(stream)
      for otherStream = 1, numStreams do
         results[stream]:add(tensors[otherStream])
      end
   end

   -- Validate that all streams received the full values
   -- As above, it would be rather hard to write a test to ensure that
   -- we're actually executing all this asynchronously, and to write a test that
   -- always guarantees failure with this race is equally problematic.
   -- So, we satisfy ourselves with this.
   for stream = 1, numStreams do
      cutorch.setStream(stream)
      tester:asserteq(results[stream]:min(), iter * numStreams)
   end

   -- return to default stream
   cutorch.setStream(0)
   results = nil
   tensors = nil
   collectgarbage()
   collectgarbage()
   cutorch.synchronize()
end

function test.streamBarrierMultiDevice()
   -- This test requires multiple devices
   local numDevices = cutorch.getDeviceCount()
   if numDevices < 2 then
      return
   end

   local size = 2000000
   local iter = 50 + torch.random(10)
   local numStreams = torch.random(10)
   cutorch.reserveStreams(numStreams)

   local tensors = {} -- per device, per stream
   local tmpResults = {} -- per device, (per other device, per other stream)
   local results = {} -- per device
   local waitingFor = {}

   -- Create space on all devices
   for device = 1, numDevices do
      cutorch.setDevice(device)
      cutorch.setStream(1)
      table.insert(results, torch.CudaTensor(size):zero())

      -- tmpResults[our device][other device][other stream]
      local tmpResultsPerDevice = {}
      for otherDevice = 1, numDevices do
         local tmpResultsPerOtherDevice = {}
         for otherStream = 1, numStreams do
            table.insert(tmpResultsPerOtherDevice, torch.CudaTensor(size):zero())
         end
         table.insert(tmpResultsPerDevice, tmpResultsPerOtherDevice)
      end
      table.insert(tmpResults, tmpResultsPerDevice)

      -- tensors[our device][our stream]
      local tensorsPerDevice = {}
      local waitingForPerDevice = {}
      for stream = 1, numStreams do
         cutorch.setStream(stream)
         table.insert(tensorsPerDevice, torch.CudaTensor(size):zero())
         table.insert(waitingForPerDevice, stream)
      end

      table.insert(tensors, tensorsPerDevice)
      table.insert(waitingFor, waitingForPerDevice)
   end

   -- Queue work to all streams, all devices
   for dev = 1, numDevices do
      cutorch.setDevice(dev)
      for stream = 1, numStreams do
         cutorch.setStream(stream)
         for i = 1, iter do
            tensors[dev][stream]:add(1)
         end
      end
   end

   -- -- Create an all-way barrier
   cutorch.streamBarrierMultiDevice(waitingFor)

   -- -- All-to-all copy (done in stream 1 on each device)
   for dev = 1, numDevices do
      cutorch.setDevice(dev)
      cutorch.setStream(1)

      for otherDev = 1, numDevices do
         for otherStream = 1, numStreams do
            tmpResults[dev][otherDev][otherStream]:copy(tensors[otherDev][otherStream])
         end
      end
   end

   -- For each device in stream 1, sum up the accumulated results from
   -- all devices/all streams
   for dev = 1, numDevices do
      cutorch.setDevice(dev)
      cutorch.setStream(1)

      for otherDev = 1, numDevices do
         for otherStream = 1, numStreams do
            results[dev]:add(tmpResults[dev][otherDev][otherStream])
         end
      end
   end

   -- Validate that all devices received the full values
   -- As above, it would be rather hard to write a test to ensure that
   -- we're actually executing all this asynchronously, and to write a test that
   -- always guarantees failure with this race is equally problematic.
   -- So, we satisfy ourselves with this.
   for dev = 1, numDevices do
      cutorch.setDevice(dev)
      cutorch.setStream(1)
      tester:asserteq(results[dev]:min(), iter * numStreams * numDevices)
   end

   -- return to default stream/device
   cutorch.setDevice(1)
   cutorch.setStream(0)
   results = nil
   tmpResults = nil
   tensors = nil
   collectgarbage()
   collectgarbage()
   cutorch.synchronize()
end

function test.cudaHostTensor()
  local t = cutorch.createCudaHostTensor(3, 4, 5)
  tester:assertTableEq(t:size():totable(), {3, 4, 5})

  local u = torch.Tensor(4, 5, 6)
  local v = cutorch.createCudaHostTensor(u:size())
  tester:assertTableEq(u:size():totable(), v:size():totable())
end

-- unfortunately, torch.Tester() forgot setUp and tearDown functions.
-- It would be nice to fix torch.Tester() eventually.
local function setUp()
  cutorch.setDevice(1)
end

for k,v in pairs(test) do
  test[k] = function()
    setUp()
    v()
  end
end

local function initSeed(seed)
   seed = seed or os.time()
   -- ensure that you can reproduce a failing test
   print('seed: ', seed)
   math.randomseed(seed)
   torch.manualSeed(seed)
   cutorch.manualSeedAll(seed)
end

function cutorch.test(tests, seed)
   initSeed(seed)
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
   os.exit(#tester.errors == 0 and 0 or 1)
end
return test
