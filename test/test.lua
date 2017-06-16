local runtests = false
if not cutorch then
   require 'cutorch'
   runtests = true
end

local test = {}
local minsize = 5
local maxsize = 10
local minvalue = 2
local maxvalue = 20
local nloop = 100
local test_tolerance = 1e-5
local unpack = unpack or table.unpack
local hasHalfChecked = false
--e.g. unit test cmd: th -lcutorch -e "cutorch.test{'view','viewAs'}"

local typenames = {
    'torch.CudaByteTensor',
    'torch.CudaCharTensor',
    'torch.CudaShortTensor',
    'torch.CudaIntTensor',
    'torch.CudaLongTensor',
    'torch.CudaTensor',
    'torch.CudaDoubleTensor'
}

local float_typenames = {
    'torch.CudaTensor',
    'torch.CudaDoubleTensor'
}

local t2gpu = {
   ['torch.ByteTensor'] = 'torch.CudaByteTensor',
   ['torch.CharTensor'] = 'torch.CudaCharTensor',
   ['torch.ShortTensor'] = 'torch.CudaShortTensor',
   ['torch.IntTensor'] = 'torch.CudaIntTensor',
   ['torch.LongTensor'] = 'torch.CudaLongTensor',
   ['torch.FloatTensor'] = 'torch.CudaTensor',
   ['torch.DoubleTensor'] = 'torch.CudaDoubleTensor',

   ['torch.ByteStorage'] = 'torch.CudaByteStorage',
   ['torch.CharStorage'] = 'torch.CudaCharStorage',
   ['torch.ShortStorage'] = 'torch.CudaShortStorage',
   ['torch.IntStorage'] = 'torch.CudaIntStorage',
   ['torch.LongStorage'] = 'torch.CudaLongStorage',
   ['torch.FloatStorage'] = 'torch.CudaStorage',
   ['torch.DoubleStorage'] = 'torch.CudaDoubleStorage',
}

local t2cpu = {}
for k,v in pairs(t2gpu) do
   t2cpu[v] = k
end

local function checkHalf()
   if cutorch.hasHalf and hasHalfChecked == false then
       table.insert(typenames, 'torch.CudaHalfTensor')
       table.insert(float_typenames, 'torch.CudaHalfTensor')
       t2cpu['torch.CudaHalfTensor'] = 'torch.FloatTensor'
       t2gpu['torch.HalfTensor'] = 'torch.CudaHalfTensor'
   end
   hasHalfChecked = true
end

local function isFloat(t)
    for k, v in pairs(float_typenames) do
        if t == k then
            return true
        end
    end
    return false
end

-- Picks an integer between a and b, inclusive of endpoints
local function chooseInt(a, b)
   return math.floor(torch.uniform(a, b + 1))
end

-- Constructs a tensor from a larger storage, with holes in each dimension
local function createHoledTensorWithSizes(size)
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

-- Create a tensor of a given size, allowing for transpositions or holes
local function createTestTensorWithSizes(allowHoles, allowTransposition, sizes)
   local t = nil
   if allowHoles then
      t = createHoledTensorWithSizes(sizes)
   else
      t = torch.FloatTensor(unpack(sizes))
   end

   if allowTransposition then
      local dims = t:nDimension()

      local numTranspositions = chooseInt(1, dims)

      for i = 1, numTranspositions do
         local dim1 = chooseInt(1, dims)
         local dim2 = chooseInt(1, dims)

         if dim1 ~= dim2 then
            t = t:transpose(dim1, dim2)
         end
      end
   end

   if allowHoles then
      -- fill the holes with NaNs (the non-holes will be overwritten below)
      -- this will help detect garbage usage
      t:storage():fill(0/0)
   end

   -- The test tensor may be used for sort/selection testing, in which
   -- case we wish to avoid duplicate elements, but might like some
   -- randomness
   t:copy(torch.randperm(t:nElement()))

   return t
end

-- Create a test tensor bounded by total size `maxSize`
local function createTestTensorMaxSize(allowHoles, allowTransposition, maxSize)
   local dims = chooseInt(1, 5)
   local maxDimSize = math.ceil(math.pow(maxSize, 1 / dims))
   local sizes = nil

   while true do
      sizes = {}
      local size = 1

      for i = 1, dims do
         sizes[i] = chooseInt(1, maxDimSize)
         size = size * sizes[i]
      end

      if (size > 1) and (size < maxSize) then
         break
      end
   end

   return createTestTensorWithSizes(allowHoles, allowTransposition, sizes)
end

-- Create a (potentially transposed, potentially with holes) tensor of a given
-- max size
local function createTestTensor(maxSize)
   -- 50/50 chance of contig/non-contig
   local contig = chooseInt(1, 2) == 1
   local holes = false
   local tr = false
   if not contig then
      holes = chooseInt(1, 2) == 1
      tr = chooseInt(1, 2) == 1
   end

   return createTestTensorMaxSize(holes, tr, maxSize)
end

local function isEqual(x, y, tolerance, ...)
   if x == nil and y == nil then return true end
   if x == nil and y ~= nil then return false end
   if x ~= nil and y == nil then return false end

   -- if x, y are tensors clone them so we can modify the contents if necessary for testing
   local a = type(x) ~= 'number' and x:clone() or x
   local b = type(y) ~= 'number' and y:clone() or y

   if torch.type(b) ~= torch.type(a) then
      b = b:typeAs(a) -- TODO: remove the need for this (a-b doesnt work for bytetensor, cudatensor pairs)
   end
   local diff = a-b
   tolerance = tolerance or 0.000001

   if type(a) == 'number' then
      -- NaN Check:
      if a ~= a and b ~= b then
          return true
      end
      return math.abs(diff) < tolerance
   else
      if torch.type(diff) ~= 'torch.FloatTensor' then
         diff = diff:float() -- TODO: remove the need for this (byteTensor and abs)
      end
      -- NaN Check:
      local hasNaN = false
      diff:apply(function(elt) if elt ~= elt then hasNaN = true end end)
      if hasNaN then
         -- check if NaN in equal positions
         local nea = torch.ne(a, a)
         local neb = torch.ne(b, b)
         if not nea:equal(neb) then
            return false
         end
         -- check diff of all other elements less than tolerance
         local ea = a:apply(function(elt) if elt ~= elt then return 0 else return elt end end)
         local eb = b:apply(function(elt) if elt ~= elt then return 0 else return elt end end)
         return (ea-eb):abs():max() < tolerance
      else
         return diff:abs():max() < tolerance
      end
   end
end

local function checkMultiDevice(x, fn, ...)
   local device_count = cutorch.getDeviceCount()
   if device_count >= 2 then
      local x = x:cuda()
      cutorch.setDevice(cutorch.getDevice() == 1 and 2 or 1)
      local ok, err = pcall(function(...) x[fn](x, ...) end, ...)
      tester:assert(not ok, "Multi-device checks failed for: " .. tostring(fn))
   end
end

local function cloneExactlyToGPU(t)
   -- keep the size/stride of original tensor, handling tensors that
   -- potentially have holes as well
   local tGPU = nil

   if t:storage() then
      local sGPU = torch.CudaStorage(t:storage():size()):copy(t:storage())
      tGPU = torch.CudaTensor(sGPU, t:storageOffset(), t:size(), t:stride())
   else
      tGPU = torch.CudaTensor()
   end

   return tGPU
end

local function compareFloatAndCuda(x, fn, ...)
   local args = {...}
   args['input'] = x
   local x_cpu = x:float()
   local x_cuda = cloneExactlyToGPU(x_cpu)

   local rcpu = {}
   local rcuda = {}
   if type(fn) == 'string' then
      tester:assertne(x_cuda[fn], nil,
		      string.format("Missing function CudaTensor.%s", fn))
      rcpu[1], rcpu[2], rcpu[3], rcpu[4] = x_cpu[fn](x_cpu, ...)
      rcuda[1], rcuda[2], rcuda[3], rcuda[4] = x_cuda[fn](x_cuda, ...)
   elseif type(fn) == 'function' then
      rcpu[1], rcpu[2], rcpu[3], rcpu[4] = fn(x_cpu, ...)
      rcuda[1], rcuda[2], rcuda[3], rcuda[4] = fn(x_cuda, ...)
   else
      error("Incorrect function type")
   end
   local errstr = string.format("Divergent results between CPU and CUDA" ..
				" for function '%s' (return value 1)", tostring(fn))
   local tolerance = test_tolerance
   tester:assert(#rcpu == #rcuda,
		 string.format("number of return arguments for CPU and CUDA "
			       .. "are different for function '%s'", tostring(fn)))
   for k, _ in ipairs(rcpu) do
      if not isEqual(rcpu[k], rcuda[k], tolerance) then
	      tester:assert(false, errstr)
      end
   end
end

local function compareFloatAndCudaTensorArgs(x, fn, ...)
   local x_cpu = x:float()
   local x_cuda = cloneExactlyToGPU(x_cpu)

   local rcpu = {}
   local rcuda = {}

   -- Transformation of args
   local tranform_args = function(t, type)
      for k,v in pairs(t) do
         local v_type = torch.Tensor.type(v)
         if v_type == 'torch.FloatTensor' or v_type == 'torch.CudaTensor'
	 or v_type == 'torch.DoubleTensor' then
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
      rcpu[1], rcpu[2], rcpu[3], rcpu[4]  = x_cpu[fn](x_cpu, unpack(cpu_args))
      rcuda[1], rcuda[2], rcuda[3], rcuda[4] = x_cuda[fn](x_cuda, unpack(cuda_args))
   elseif type(fn) == 'function' then
      rcpu[1], rcpu[2], rcpu[3], rcpu[4] = fn(x_cpu, unpack(cpu_args))
      rcuda[1], rcuda[2], rcuda[3], rcuda[4] = fn(x_cuda, unpack(cuda_args))
   else
      error("Incorrect function type")
   end
   local errstr = string.format("Divergent results between CPU and CUDA" ..
				" for function '%s' (return value 1)", tostring(fn))
   local tolerance = test_tolerance
   tester:assert(#rcpu == #rcuda,
		 string.format("number of return arguments for CPU and CUDA "
			       .. "are different for function '%s'", tostring(fn)))
   for k, _ in ipairs(rcpu) do
      if not isEqual(rcpu[k], rcuda[k], tolerance) then
	 print(args)
	 tester:assert(false, errstr)
      end
   end
end

-- converts a tensor to it's exact GPU type
local function GPU(t, gpu2cpu_map)
   gpu2cpu_map = gpu2cpu_map or t2gpu
   if torch.isTensor(t) or torch.isStorage(t) then
      return torch[gpu2cpu_map[torch.type(t)]:match('torch.(%a+)')] or t
   elseif torch.type(t) == 'string' then
      return torch[gpu2cpu_map[t]:match('torch.(%a+)')]
   end
   error('not tensor or storage')
end

-- converts a tensor to it's exact CPU type
local function CPU(t)
   if torch.isTensor(t) or torch.isStorage(t) then
      return torch[t2cpu[torch.type(t)]:match('torch.(%a+)')] or t
   elseif torch.type(t) == 'string' then
      return torch[t2cpu[t]:match('torch.(%a+)')]
   end
   error('not tensor or storage')
end

-- exactly clone a tensor (same size / storage) to it's equivalent GPU type
-- if baseType is given, convert to the baseType's GPU type instead
local function cloneExactlyToGPUType(t, baseType, gpu2cpu_map)
   local type = baseType and baseType or t
   -- keep the size/stride of original tensor, handling tensors that
   -- potentially have holes as well
   local tGPU = nil
   if t:storage() then
      local sGPU = GPU(type, gpu2cpu_map).new(1):storage().new(t:storage():size()):copy(t:storage())
      tGPU = GPU(type, gpu2cpu_map)(sGPU, t:storageOffset(), t:size(), t:stride())
   else
      tGPU = GPU(type, gpu2cpu_map)()
   end

   return tGPU
end

-- baseType = the tensor type to test
-- indexMode = true: keep indexing and masking Tensors as their CPU equivalents
--             false: convert then to baseType when doing CUDA
-- x = first argument tensor
-- limit: number of returns to compare, if nil, compares all returns
-- gpu2cpu_map = map of gpu types to cpu types
-- fn = function name (as string), or the function)
-- ... = the rest of arguments to fn
local function compareCPUAndCUDATypeTensorArgsWithConvInternal(cudaType, gpu2cpu_map, indexMode, limit, x, fn, ...)
   local baseType = t2cpu[cudaType]
   assert(baseType, 'Cannot find baseType for ' .. cudaType)
   local x_cpu = x:type(baseType)
   local x_cuda = cloneExactlyToGPUType(x_cpu, nil, gpu2cpu_map)

   local rcpu = {}
   local rcuda = {}
   -- Transformation of args
   local tranform_args = function(t, type)
      for k,v in pairs(t) do
	 if torch.isTensor(v) or torch.isStorage(v) then
	    if indexMode == true then
                t[k] = cloneExactlyToGPUType(v, nil, gpu2cpu_map)
	    else
                t[k] = cloneExactlyToGPUType(v, x_cpu, gpu2cpu_map)
	    end
         end
      end
      return t
   end

   local cpu_args = {...}
   local cuda_args = tranform_args({...})
   if type(fn) == 'string' then
      tester:assertne(x_cuda[fn], nil,
                     string.format("Missing function %s.%s", torch.type(x_cuda), fn))
      rcpu[1], rcpu[2], rcpu[3], rcpu[4]  = x_cpu[fn](x_cpu, unpack(cpu_args))
      rcuda[1], rcuda[2], rcuda[3], rcuda[4] = x_cuda[fn](x_cuda, unpack(cuda_args))
   elseif type(fn) == 'function' then
      rcpu[1], rcpu[2], rcpu[3], rcpu[4] = fn(x_cpu, unpack(cpu_args))
      rcuda[1], rcuda[2], rcuda[3], rcuda[4] = fn(x_cuda, unpack(cuda_args))
   else
      error("Incorrect function type")
   end

   local tolerance = test_tolerance
   local errstr = string.format("Divergent results between CPU and CUDA"
				.. " for function '%s.%s", torch.type(x_cuda), fn)
   if indexMode ~= nil then
      errstr = errstr .. " in indexMode = " .. tostring(indexMode)
   end
   errstrval = errstr .. " for return value # %d"
   errstrval = errstrval .. ". Divergence value: %f"
   errstrobj = errstr .. " for object"
   errstrobj = errstrobj .. ". Divergence value: %f"
   local function divval(cpu, cuda)
      return torch.isTensor(cpu) and (cpu:double() - cuda:double()):abs():max() or 0
   end

   tester:assert(#rcpu == #rcuda,
		 string.format("number of return arguments for CPU and CUDA "
			       .. "are different for function '%s'", tostring(fn)))

   if limit ~= nil then
      for k = 1, limit do
         tester:assert(isEqual(rcpu[k], rcuda[k], tolerance),
                       string.format(errstrval, k, divval(rcpu[k], rcuda[k])))
      end
   else
      for k, _ in ipairs(rcpu) do
         tester:assert(isEqual(rcpu[k], rcuda[k], tolerance),
                       string.format(errstrval, k, divval(rcpu[k], rcuda[k])))
      end
   end

   -- also test x in case function changed object
   tester:assert(isEqual(x_cpu, x_cuda, tolerance),
                 string.format(errstrobj, divval(x_cpu, x_cuda)))
end

local function compareCPUAndCUDATypeTensorArgs(cudaType, indexMode, x, fn, ...)
   compareCPUAndCUDATypeTensorArgsWithConvInternal(cudaType, nil, indexMode, nil, x, fn, ...)
end

local function compareCPUAndCUDATypeTensorArgsWithLimit(cudaType, indexMode, limit, x, fn, ...)
   compareCPUAndCUDATypeTensorArgsWithConvInternal(cudaType, nil, indexMode, limit, x, fn, ...)
end

function test.squeeze()
   local sz = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz, 1, sz, 1)
   for k, typename in ipairs(typenames) do
      local x = x:type(typename)
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'squeeze')
   end

   local y = x:cuda():squeeze()
   tester:assert(y:dim() == 2, "squeeze err")

   x = torch.FloatTensor():rand(sz, 1, 1, sz)
   for k, typename in ipairs(typenames) do
      local x = x:type(typename)
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'squeeze', 2)
   end

   local y = x:cuda():squeeze(2)
   tester:assert(y:dim() == 3, "squeeze1d err")

   x = torch.FloatTensor(1):normal()
   for k, typename in ipairs(typenames) do
      local x = x:type(typename)
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'squeeze')
   end
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
   tester:asserteq(nelem1, nelem2, 'input and output sizes have to be the same')
   local input, output

   -- extract a sub-cube with probability 50%
   -- (to introduce unreachable storage locations)
   local holedInput = torch.random(1, 2)
   local holedOutput = torch.random(1, 2)
   if holedInput == 1 then
      input = createHoledTensorWithSizes(inputSize)
   else
      input = torch.FloatTensor(torch.LongStorage(inputSize))
   end
   input:storage():fill(-150)
   input:copy(torch.linspace(1, input:nElement(), input:nElement()))

   if holedOutput == 1 then
      output = createHoledTensorWithSizes(outputSize)
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

   -- Also test cross-device copy behavior, if multiple devices are available.
   local input_device = chooseInt(1, cutorch.getDeviceCount())
   local output_device = chooseInt(1, cutorch.getDeviceCount())

   -- Selectively disable p2p access to test that codepath as well
   local access_disabled = false
   if input_device ~= output_device and chooseInt(1, 2) == 1 then
      -- p2p access between this pair of devices might not be available at all
      if cutorch.getPeerToPeerAccess(output_device, input_device) then
         access_disabled = true
         cutorch.setPeerToPeerAccess(output_device, input_device, false)
      end
   end

   local prev_device = cutorch.getDevice()

   cutorch.setDevice(input_device)
   local input_tensor_cuda = torch.CudaTensor(input_storage_cuda,
                                          input_tensor_float:storageOffset(),
                                          input_tensor_float:size(),
                                          input_tensor_float:stride())

   cutorch.setDevice(output_device)
   local output_tensor_cuda = torch.CudaTensor(output_storage_cuda,
                                          output_tensor_float:storageOffset(),
                                          output_tensor_float:size(),
                                          output_tensor_float:stride())

   cutorch.setDevice(prev_device)

   output_tensor_float:copy(input_tensor_float)
   output_tensor_cuda:copy(input_tensor_cuda)

   if access_disabled then
      cutorch.setPeerToPeerAccess(output_device, input_device, true)
   end

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

function test.copyAsync()
   local sz = chooseInt(maxsize, 2 * maxsize)
   local host_tensors = {
     cutorch.createCudaHostTensor(sz),
     cutorch.createCudaHostDoubleTensor(sz)
   }
   if cutorch.hasHalf then
     table.insert(host_tensors, cutorch.createCudaHostHalfTensor(sz))
   end
   for k,host_tensor in ipairs(host_tensors) do
      local device_type = t2gpu[torch.type(host_tensor)]:match(('torch.(%a+)'))
      if torch.type(host_tensor) ~= 'torch.HalfTensor' then
         host_tensor = host_tensor:uniform()
      else
         -- HalfTensor doesn't have math functions defined.
         local copy_tensor = torch[device_type](sz):uniform()
         host_tensor:copy(copy_tensor)
      end
      local device_tensor = torch[device_type](sz)
      device_tensor:copyAsync(host_tensor)
      cutorch.streamSynchronize(cutorch.getStream())
      tester:assertTensorEq(host_tensor:double(), device_tensor:double(), 0,
                            "Async copy to device failed.")

      device_tensor:uniform()
      host_tensor:copyAsync(device_tensor)
      cutorch.streamSynchronize(cutorch.getStream())
      tester:assertTensorEq(device_tensor:double(), host_tensor:double(), 0,
                            "Async copy to host failed.")
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
   for k, typename in ipairs(typenames) do
       local x = x:type(t2cpu[typename])
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'zero')
   end
   checkMultiDevice(x, 'zero')
end

function test.fill()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local v = torch.uniform()
   for k, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'fill', v)
   end
   checkMultiDevice(x, 'fill', v)
end

function test.reshape()
   local sz1 = chooseInt(minsize, maxsize)*2
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   for k, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'reshape', sz1/2, sz2*2)
   end
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

function test.linspace()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local n = sz1 * sz2
   local a = torch.uniform()
   local b = torch.uniform()
   local x = torch.FloatTensor():rand(sz1, sz2)
   for k, typename in ipairs(float_typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'linspace', a, b, n)
   end
   checkMultiDevice(x, 'linspace', a, b, n)

   -- Check range for non-contiguous tensors.
   local x = createTestTensorWithSizes(true, true, {sz1, sz2})
   for k, typename in ipairs(float_typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'linspace', a, b, n)
   end
   checkMultiDevice(x, 'linspace', a, b, n)

   -- Ckeck new tensor creation
   local x = torch.FloatTensor()
   for k, typename in ipairs(float_typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'linspace', a, b, n)
   end
   checkMultiDevice(x, 'linspace', a, b, n)

   -- Ckeck n = 1 case
   local x = torch.rand(1)
   for k, typename in ipairs(float_typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'linspace', a, a, 1)
   end
   checkMultiDevice(x, 'linspace', a, a, 1)

   -- Ckeck default parameter case
   local x = createTestTensorWithSizes(true, true, {100})
   for k, typename in ipairs(float_typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'linspace', a, b)
   end
   checkMultiDevice(x, 'linspace', a, b)
end

function test.logspace()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local n = sz1 * sz2
   local a = torch.uniform()
   local b = torch.uniform()
   local x = torch.FloatTensor():rand(sz1, sz2)
   for k, typename in ipairs(float_typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'logspace', a, b, n)
   end
   checkMultiDevice(x, 'logspace', a, b, n)

   -- Check range for non-contiguous tensors.
   local x = createTestTensorWithSizes(true, true, {sz1, sz2})
   for k, typename in ipairs(float_typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'logspace', a, b, n)
   end
   checkMultiDevice(x, 'logspace', a, b, n)

   -- Ckeck new tensor creation
   local x = torch.FloatTensor()
   for k, typename in ipairs(float_typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'logspace', a, b, n)
   end
   checkMultiDevice(x, 'logspace', a, b, n)

   -- Ckeck n = 1 case
   local x = torch.rand(1)
   for k, typename in ipairs(float_typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'logspace', a, a, 1)
   end
   checkMultiDevice(x, 'logspace', a, a, 1)

   -- Ckeck default parameter case
   local x = createTestTensorWithSizes(true, true, {100})
   for k, typename in ipairs(float_typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'logspace', a, b)
   end
   checkMultiDevice(x, 'logspace3', a, b)
end


function test.add()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor():rand(sz1, sz2)
   local v = torch.uniform()
   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local x, y, z = x:type(ctype), y:type(ctype), z:type(ctype)
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'add', z)
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'add', z, v)
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'add', y, z)
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'add', y, v, z)
   end
   checkMultiDevice(x, 'add', z)
   checkMultiDevice(x, 'add', z, v)
   checkMultiDevice(x, 'add', y, z)
   checkMultiDevice(x, 'add', y, v, z)
end

local test_bitops =  function(funcname, tmin, tmax, vmin, vmax)
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.IntTensor(sz1, sz2):random(tmin, tmax)
   local v = torch.random(vmin, vmax)
   compareCPUAndCUDATypeTensorArgs('torch.CudaIntTensor', nil, x, funcname, v)
   checkMultiDevice(x, funcname, v)
end

function test.lshift()
   test_bitops('lshift', 1, 1000, 1, 10)
end

function test.rshift()
   test_bitops('rshift', 1000, 1000000, 1, 10)
end

function test.bitand()
   test_bitops('bitand', 1, 1000, 1, 255)
end

function test.bitor()
   test_bitops('bitor', 1, 1000, 1, 255)
end

function test.bitxor()
   test_bitops('bitxor', 1, 1000, 1, 255)
end

function test.csub()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor():rand(sz1, sz2)
   local v = torch.uniform()
   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local x, y, z = x:type(ctype), y:type(ctype), z:type(ctype)
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'csub', z)
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'csub', z, v)
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'csub', y, z)
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'csub', y, v, z)
   end
   checkMultiDevice(x, 'csub', z)
   checkMultiDevice(x, 'csub', z, v)
   checkMultiDevice(x, 'csub', y, z)
   checkMultiDevice(x, 'csub', y, v, z)
end

function test.cmul()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   for k, typename in ipairs(typenames) do
       local ctype = t2cpu[typename]
       local x, y = x:type(ctype), y:type(ctype)
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'cmul', y)
   end
   checkMultiDevice(x, 'cmul', y)
end

function test.cpow()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   for k, typename in ipairs(typenames) do
       local ctype = t2cpu[typename]
       local x, y = x:type(ctype), y:type(ctype)
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'cpow', y)
   end
   checkMultiDevice(x, 'cpow', y)
end

function test.cremainder()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor(sz1, sz2):uniform(-50, 50)
   local y = torch.FloatTensor(sz1, sz2):uniform(-50, 50)
   for k, typename in ipairs(typenames) do
       local ctype = t2cpu[typename]
       local a, b = x:type(ctype), y:type(ctype)
       if not isFloat(typename) then
           b[b:eq(0)] = 1
       end
       compareCPUAndCUDATypeTensorArgs(typename, nil, a, 'cremainder', b)
   end
   checkMultiDevice(x, 'cremainder', y)

   -- ensure we test divide by zero
   local x = torch.FloatTensor(1):fill(1)
   local y = torch.FloatTensor(1):zero()
   for k, typename in ipairs(float_typenames) do
       local ctype = t2cpu[typename]
       local a, b = x:type(ctype), y:type(ctype)
       compareCPUAndCUDATypeTensorArgs(typename, nil, a, 'cremainder', b)
   end
   checkMultiDevice(x, 'cremainder', y)
end

function test.cfmod()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor(sz1, sz2):uniform(-50, 50)
   local y = torch.FloatTensor(sz1, sz2):uniform(-50, 50)
   for k, typename in ipairs(typenames) do
       local ctype = t2cpu[typename]
       local a, b = x:type(ctype), y:type(ctype)
       if not isFloat(typename) then
           b[b:eq(0)] = 1
       end
       compareCPUAndCUDATypeTensorArgs(typename, nil, a, 'cfmod', b)
   end
   checkMultiDevice(x, 'cfmod', y)

   -- ensure we test mod by zero
   local x = torch.FloatTensor(1):fill(1)
   local y = torch.FloatTensor(1):zero()
   for k, typename in ipairs(float_typenames) do
       local ctype = t2cpu[typename]
       local a, b = x:type(ctype), y:type(ctype)
       compareCPUAndCUDATypeTensorArgs(typename, nil, a, 'cfmod', b)
   end
   checkMultiDevice(x, 'cfmod', y)
end

function test.nonzero()
    local minsize = 10
    local maxsize = 20
    local dims = {chooseInt(minsize, maxsize)}
    local threshold = 1 / 3
    local flip = math.random()
    while flip > threshold do
        dims[#dims + 1] = chooseInt(minsize, maxsize)
        flip = math.random()
    end
    local x = createTestTensorWithSizes(true, true, dims)
    local randMask = torch.ByteTensor(unpack(dims)):bernoulli()
    x:maskedFill(randMask, 0)
    for k, typename in ipairs(typenames) do
        local ctype = t2cpu[typename]
        local x = x:type(ctype)
        compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'nonzero')
    end
    checkMultiDevice(x, 'nonzero')
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

   for _, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      local y = y:type(t2cpu[typename])
      local z = z:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'addcmul', y, z)
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'addcmul', torch.uniform(), y, z)
   end

   checkMultiDevice(x, 'addcmul', y, z)
   checkMultiDevice(x, 'addcmul', torch.uniform(), y, z)

   local r = torch.zeros(sz1, sz2)
   for _, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      local y = y:type(t2cpu[typename])
      local z = z:type(t2cpu[typename])
      local r = r:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, r, 'addcmul', x, y, z)
      compareCPUAndCUDATypeTensorArgs(typename, nil, r, 'addcmul', x, torch.uniform(), y, z)
   end

   checkMultiDevice(r, 'addcmul', x, y, z)
   checkMultiDevice(r, 'addcmul', x, torch.uniform(), y, z)

end

function test.addcdiv()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   -- add so no divide by zero
   local x = torch.FloatTensor():rand(sz1, sz2):add(torch.random(1, 5))
   local y = torch.FloatTensor():rand(sz1, sz2):add(torch.random(1, 5))
   local z = torch.FloatTensor():rand(sz1, sz2):add(torch.random(1, 5))

   for _, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      local y = y:type(t2cpu[typename])
      local z = z:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'addcdiv', y, z)
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'addcdiv', torch.uniform(), y, z)
   end

   checkMultiDevice(x, 'addcdiv', y, z)
   checkMultiDevice(x, 'addcdiv', torch.uniform(), y, z)

   local r = torch.zeros(sz1, sz2)
   for _, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      local y = y:type(t2cpu[typename])
      local z = z:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, r, 'addcdiv', x, y, z)
      compareCPUAndCUDATypeTensorArgs(typename, nil, r, 'addcdiv', x, torch.uniform(), y, z)
   end

   checkMultiDevice(r, 'addcdiv', x, y, z)
   checkMultiDevice(r, 'addcdiv', x, torch.uniform(), y, z)
end

function test.fmod()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():randn(sz1, sz2)
   x:apply(function(x)
       x = x * torch.random(1, 100)
       return x
   end)
   local r = torch.normal(0, 25)

   for _, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'fmod', r)
   end
end

function test.remainder()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():randn(sz1, sz2)
   x:apply(function(x)
       x = x * torch.random(1, 100)
       return x
   end)
   local r = torch.normal(0, 25)

   for _, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'remainder', r)
   end
end

function test.equal()
    -- empty tensors are equal
    local x = torch.FloatTensor()
    local y = torch.FloatTensor()

    for _, typename in ipairs(typenames) do
        local a = x:type(typename)
        local b = y:type(typename)
        tester:assert(a:equal(b), 'Empty Tensors should be considered equal')
    end

    -- mismatched size tensors are not equal
    local x = torch.FloatTensor(5):fill(1)
    local y = torch.FloatTensor(3):fill(1)

    for _, typename in ipairs(typenames) do
        local a = x:type(typename)
        local b = y:type(typename)
        tester:assert(not a:equal(b), 'Tensors of different sizes not equal')
    end

    -- tensors of same size but different value are not equal
    local sz1 = chooseInt(minsize, maxsize)
    local sz2 = chooseInt(minsize, maxsize)
    local x = torch.FloatTensor(sz1, sz2):apply(function() return torch.random(0, 255) end)
    local y = torch.add(x, 1)

    for _, typename in ipairs(typenames) do
        local a = x:type(typename)
        local b = y:type(typename)
        tester:assert(not a:equal(b), 'Tensors should not be equal')
    end

    -- actual equality
    for _, typename in ipairs(typenames) do
        local a = x:type(typename)
        local b = x:type(typename)
        tester:assert(a:equal(b), 'Tensors should be equal')
    end
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
   for k, typename in ipairs(typenames) do
     local x = x:type(t2cpu[typename])
     compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'zero')
   end
   checkMultiDevice(x, 'mean')
   checkMultiDevice(x, 'mean', 1)
end

function test.max()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.randperm(sz1 * sz2):view(sz1, sz2):float()
   for k, typename in ipairs(typenames) do
      local x_
      if typename == 'torch.CudaByteTensor' or typename == 'torch.CudaCharTensor'
      or typename == 'torch.CudaShortTensor' then
	 -- limit the range of max, so that there's no same indices
	 local sz1 = chooseInt(1, 10)
	 local sz2 = chooseInt(1, 10)
	 x_ = torch.randperm(sz1 * sz2):view(sz1, sz2)
      else
         x_ = x:type(t2cpu[typename])
      end
      compareCPUAndCUDATypeTensorArgs(typename, nil, x_, 'max')
      compareCPUAndCUDATypeTensorArgs(typename, nil, x_, 'max', 1)
      compareCPUAndCUDATypeTensorArgs(typename, nil, x_, 'max', 2)
   end
   checkMultiDevice(x, 'max')
   checkMultiDevice(x, 'max', 1)
end

function test.min()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.randperm(sz1 * sz2):view(sz1, sz2):float()
   for k, typename in ipairs(typenames) do
      local x_
      if typename == 'torch.CudaByteTensor' or typename == 'torch.CudaCharTensor'
      or typename == 'torch.CudaShortTensor' then
	 -- limit the range of min, so that there's no same indices
	 local sz1 = chooseInt(1, 10)
	 local sz2 = chooseInt(1, 10)
	 x_ = torch.randperm(sz1 * sz2):view(sz1, sz2)
      else
         x_ = x:type(t2cpu[typename])
      end
      compareCPUAndCUDATypeTensorArgs(typename, nil, x_, 'min')
      compareCPUAndCUDATypeTensorArgs(typename, nil, x_, 'min', 1)
      compareCPUAndCUDATypeTensorArgs(typename, nil, x_, 'min', 2)
   end
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

  for _, typename in ipairs(typenames) do
      local a = a:type(t2cpu[typename])
      local b = b:type(t2cpu[typename])
      local c = c:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, c, 'cmax', a, b)
      compareCPUAndCUDATypeTensorArgs(typename, nil, c, 'cmax', a, v)
      compareCPUAndCUDATypeTensorArgs(typename, nil, a, 'cmax', b)
      compareCPUAndCUDATypeTensorArgs(typename, nil, a, 'cmax', v)
  end

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

  for _, typename in ipairs(typenames) do
      local a = a:type(t2cpu[typename])
      local b = b:type(t2cpu[typename])
      local c = c:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, c, 'cmin', a, b)
      compareCPUAndCUDATypeTensorArgs(typename, nil, c, 'cmin', a, v)
      compareCPUAndCUDATypeTensorArgs(typename, nil, a, 'cmin', b)
      compareCPUAndCUDATypeTensorArgs(typename, nil, a, 'cmin', v)
  end

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
         t = torch.CudaByteTensor(size1):fill(1)
      else
         local size2 = chooseInt(10, 100)
         t = torch.CudaByteTensor(size1, size2):fill(1)

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
   for _, typename in ipairs(typenames) do
       local x = x:type(t2cpu[typename])
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'cumsum');
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'cumsum', 1);
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'cumsum', 2);
   end
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
   for _, typename in ipairs(typenames) do
       local x = x:type(t2cpu[typename])
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'cumprod');
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'cumprod', 1);
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'cumprod', 2);
   end
   checkMultiDevice(x, 'cumprod')
   checkMultiDevice(x, 'cumprod', 1)
end

function test.var()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)

   for _, typename in ipairs(float_typenames) do
     local x = x:type(t2cpu[typename])
     compareFloatAndCuda(x, 'var')
     compareFloatAndCuda(x, 'var', 1, true)
     compareFloatAndCuda(x, 'var', 1, false)
     compareFloatAndCuda(x, 'var', 2, true)
     compareFloatAndCuda(x, 'var', 2, false)
   end

   checkMultiDevice(x, 'var')
   checkMultiDevice(x, 'var', 1)
end

function test.std()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)

   for _, typename in ipairs(float_typenames) do
     local x = x:type(t2cpu[typename])
     compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'std')
     compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'std', 1, true)
     compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'std', 1, false)
     compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'std', 2, true)
     compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'std', 2, false)
   end

   checkMultiDevice(x, 'std')
   checkMultiDevice(x, 'std', 1)
end

function test.diag()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local k = chooseInt(-minsize, minsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   for _, typename in ipairs(float_typenames) do
       local x = x:type(t2cpu[typename])
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'diag')
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'diag', k)
   end
   checkMultiDevice(x, 'diag')
   checkMultiDevice(x, 'diag', k)

   local y = torch.FloatTensor():rand(sz1)
   for _, typename in ipairs(float_typenames) do
       local y = x:type(t2cpu[typename])
       compareCPUAndCUDATypeTensorArgs(typename, nil, y, 'diag')
       compareCPUAndCUDATypeTensorArgs(typename, nil, y, 'diag', k)
   end
   checkMultiDevice(y, 'diag')
   checkMultiDevice(y, 'diag', k)

   -- test non-contiguous cases
   local x1 = createTestTensorWithSizes(true, true, {sz1, sz2});
   for _, typename in ipairs(float_typenames) do
       local x1 = x1:type(t2cpu[typename])
       compareCPUAndCUDATypeTensorArgs(typename, nil, x1, 'diag')
       compareCPUAndCUDATypeTensorArgs(typename, nil, x1, 'diag', k)
   end
   checkMultiDevice(x1, 'diag')
   checkMultiDevice(x1, 'diag', k)

   local y1 = createTestTensorWithSizes(true, true, {sz1});
   for _, typename in ipairs(float_typenames) do
       local y1 = y1:type(t2cpu[typename])
       compareCPUAndCUDATypeTensorArgs(typename, nil, y1, 'diag')
       compareCPUAndCUDATypeTensorArgs(typename, nil, y1, 'diag', k)
   end
   checkMultiDevice(y1, 'diag')
   checkMultiDevice(y1, 'diag', k)
end

function test.range()
   local xmin = chooseInt(minsize, maxsize)
   local xmax = chooseInt(xmin, maxsize)
   local step = 3
   local size = math.floor((xmax - xmin) / step + 1)
   -- Base case
   local x = torch.FloatTensor():rand(size)
   for k, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'range', xmin, xmax, step)
   end
   checkMultiDevice(x, 'range', xmin, xmax, step)

   -- Check range for non-contiguous tensors.
   local x = createTestTensorWithSizes(true, true, {size})
   for k, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'range', xmin, xmax, step)
   end
   checkMultiDevice(x, 'range', xmin, xmax, step)

   -- Ckeck new tensor creation
   local x = torch.Tensor()
   for k, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'range', xmin, xmax, step)
   end
   checkMultiDevice(x, 'range', xmin, xmax, step)

   -- Ckeck negative step case
   for k, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'range', xmax, xmin, -step)
   end
   checkMultiDevice(x, 'range', xmax, xmin, -step)

   -- Ckeck default parameter case
   local x = createTestTensorWithSizes(true, true, {size})
   for k, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'range', xmin, xmax)
   end
   checkMultiDevice(x, 'range', xmin, xmax, step)

   -- Ckeck floating step case
   local step = 1.3
   local x = torch.Tensor()
   for k, typename in ipairs(float_typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'range', xmin, xmax)
   end
   checkMultiDevice(x, 'range', xmin, xmax, step)
end

function test.trace()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   for _, typename in ipairs(float_typenames) do
       local x = x:type(t2cpu[typename])
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'trace')
   end
   checkMultiDevice(x, 'trace')
end

function test.tril()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   for _, typename in ipairs(float_typenames) do
       local x = x:type(t2cpu[typename])
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'tril')
   end
   checkMultiDevice(x, 'tril')
end

function test.triu()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   for _, typename in ipairs(float_typenames) do
       local x = x:type(t2cpu[typename])
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'triu')
   end
   checkMultiDevice(x, 'triu')
end

-- Test element-wise unary operators with both one and two arguments.
local function testUnary1(fnp, types, tensor)
   local fn = fnp[1]
   local min = fnp[2]
   local max = fnp[3]
   local function test()
      local sz1 = chooseInt(minsize, maxsize)
      local sz2 = chooseInt(minsize, maxsize)
      local x = tensor and tensor or torch.DoubleTensor(sz1, sz2):uniform(min, max)
      for k, typename in ipairs(types and types or float_typenames) do
         local x = x:type(t2cpu[typename]):clone()
         compareCPUAndCUDATypeTensorArgs(typename, nil, x, fn)
      end
   end
   return test
end

local function testUnary2(fnp, types)
   local fn = fnp[1]
   local min = fnp[2]
   local max = fnp[3]
   local function test()
      local sz1 = chooseInt(minsize, maxsize)
      local sz2 = chooseInt(minsize, maxsize)
      local x = torch.DoubleTensor(sz1, sz2):uniform(min, max)
      local y = torch.DoubleTensor()
      for k, typename in ipairs(types and types or float_typenames) do
          local x = x:type(t2cpu[typename]):clone()
          local y = y:type(t2cpu[typename]):clone()
         compareCPUAndCUDATypeTensorArgs(typename, nil, y, fn, x)
      end
      checkMultiDevice(y, fn, x)
   end
   return test
end

for _,name in ipairs({
      {"log", 0.001, 2},
      {"log1p", -0.9, 2},
      {"exp", -2, 2},
      {"cos", -2, 2},
      {"acos", -1, 1},
      {"cosh", -2, 2},
      {"sin", -2, 2},
      {"asin", -1, 1},
      {"sinh", -2, 2},
      {"tan", -2, 2},
      {"atan", -2, 2},
      {"tanh", -2, 2},
      {"sqrt", 0, 2},
      {"neg", -100, 100},
      {"sigmoid", -2, 2},
      {"ceil", -100, 100},
      {"floor", -100, 100},
      {"frac", -100, 100},
      {"trunc", -100, 100},
      {"cinv", -2, 2},
      {"round", -100, 100}}) do

   test[name[1] .. "1"] = testUnary1(name)
   test[name[1] .. "2"] = testUnary2(name)

end

test["abs1"] = testUnary1({"abs", -100, 100}, {'torch.CudaIntTensor',
                                               'torch.CudaLongTensor'})
test["abs2"] = testUnary2({"abs", -100, 100}, {'torch.CudaIntTensor',
                                               'torch.CudaLongTensor'})


test["sign1"] = testUnary1({"sign", -100, 100}, typenames)
test["sign2"] = testUnary2({"sign", -100, 100}, typenames)
test["sign3"] = testUnary1({"sign", -100, 100}, typenames, torch.ByteTensor(10):fill(0))

function test.rsqrt()
   local old_tolerance = test_tolerance
   test_tolerance = 1E-1  -- max observed error with 500x500 tensors in 10000 runs was 0.01157
   testUnary1('rsqrt')
   testUnary2('rsqrt')
   test_tolerance = old_tolerance
end

function test.atan2()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local z = torch.FloatTensor()
   compareFloatAndCudaTensorArgs(z, 'atan2', x, y)
   checkMultiDevice(z, 'atan2', x, y)
end

function test.lerp()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   local w = math.random()
   local z = torch.FloatTensor()
   for _, typename in ipairs(float_typenames) do
       local x = x:type(t2cpu[typename])
       local y = y:type(t2cpu[typename])
       local z = z:type(t2cpu[typename])
       compareCPUAndCUDATypeTensorArgs(typename, nil, z, 'lerp', x, y, w)
   end
   checkMultiDevice(z, 'lerp', x, y, w)
end

function test.pow1()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local pow = torch.uniform(minvalue,maxvalue)
   for k, typename in ipairs(float_typenames) do
       local ctype = t2cpu[typename]
       local x = x:type(ctype)
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'pow', pow)
   end
   checkMultiDevice(x, 'pow', pow)
end

function test.pow2()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor()
   local pow = torch.uniform(minvalue,maxvalue)
   for k, typename in ipairs(float_typenames) do
       local ctype = t2cpu[typename]
       local x, y = x:type(ctype), y:type(ctype)
       compareCPUAndCUDATypeTensorArgs(typename, nil, y, 'pow', x, pow)
   end
   checkMultiDevice(y, 'pow', x, pow)
end

function test.powExponentTensor()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local pow = torch.uniform(minvalue,maxvalue)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor()
   for k, typename in ipairs(float_typenames) do
       local ctype = t2cpu[typename]
       local x, y = x:type(ctype), y:type(ctype)
       compareCPUAndCUDATypeTensorArgs(typename, nil, y, 'pow', pow, x)
   end
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
   for _, typename in ipairs(typenames) do
      if typename ~= 'torch.CudaCharTensor' and typename ~= 'torch.CudaByteTensor' then
        local x = x:type(t2cpu[typename])
        compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'clamp', min_val, max_val);
      end
   end
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
   for _, typename in ipairs(typenames) do
      if typename ~= 'torch.CudaCharTensor' and typename ~= 'torch.CudaByteTensor' then
        local x = x:type(t2cpu[typename])
        local y = y:type(t2cpu[typename])
        compareCPUAndCUDATypeTensorArgs(typename, nil, y, 'clamp', x, min_val, max_val);
      end
   end
   checkMultiDevice(y, 'clamp', x, min_val, max_val)
end

-- same as clamp1, clamp2 but only allow positive values
function test.clamp3()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2):mul(5);
   local min_val = 1
   local max_val = 3
   x[1][1] = min_val - 1
   if sz2 >= 2 then
     x[1][2] = max_val + 1
   end
   for _, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'clamp', min_val, max_val);
   end
   checkMultiDevice(x, 'clamp', min_val, max_val)
end

function test.clamp4()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2):mul(5);
   local min_val = 1
   local max_val = 3
   x[1][1] = min_val - 1
   if sz2 >= 2 then
     x[1][2] = max_val + 1
   end
   local y = torch.FloatTensor():resizeAs(x)
   for _, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      local y = y:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, y, 'clamp', x, min_val, max_val);
   end
   checkMultiDevice(x, 'clamp', min_val, max_val)
end

function test.index()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local sz3 = chooseInt(10, 20)
   local x = torch.FloatTensor():rand(sz1, sz2)

   local longIndex = torch.LongTensor{chooseInt(1, sz1), chooseInt(1, sz1)}
   local index = 1
   for k, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, true, x, 'index',
                                      index, longIndex)
      if typename ~= 'torch.CudaByteTensor' and typename ~= 'torch.CudaCharTensor' then
          compareCPUAndCUDATypeTensorArgs(typename, false, x, 'index',
                                          index, longIndex)
      end
   end

   index = 2
   longIndex =  torch.LongTensor{chooseInt(1, sz2), chooseInt(1, sz2)}
   for k, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, true, x, 'index',
                                      index, longIndex)
      if typename ~= 'torch.CudaByteTensor' and typename ~= 'torch.CudaCharTensor' then
          compareCPUAndCUDATypeTensorArgs(typename, false, x, 'index',
                                          index, longIndex)
      end
   end

   x = torch.FloatTensor():rand(sz1)
   index = 1
   longIndex = torch.LongTensor{chooseInt(1, sz1), chooseInt(1, sz1)}
   for k, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, true, x, 'index',
                                      index, longIndex)
      if typename ~= 'torch.CudaByteTensor' and typename ~= 'torch.CudaCharTensor' then
          compareCPUAndCUDATypeTensorArgs(typename, false, x, 'index',
                                          index, longIndex)
      end
   end

   x = torch.FloatTensor():rand(sz1,sz2,sz3)
   index = 3
   longIndex = torch.randperm(sz3):long()
   for k, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, true, x, 'index',
                                      index, longIndex)
      if typename ~= 'torch.CudaByteTensor' and typename ~= 'torch.CudaCharTensor' then
          compareCPUAndCUDATypeTensorArgs(typename, false, x, 'index',
                                          index, longIndex)
      end
   end

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
   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local x, src = x:type(ctype), src:type(ctype)
      compareCPUAndCUDATypeTensorArgs(typename, true, x, 'indexCopy',
                                      index, longIndex, src)
      if typename ~= 'torch.CudaByteTensor' and typename ~= 'torch.CudaCharTensor' then
          compareCPUAndCUDATypeTensorArgs(typename, false, x, 'indexCopy',
                                          index, longIndex, src)
      end
   end

   -- Case 2: 2D tensor, indexCopy over second dimension, 2 indices
   index = 2
   longIndex =  torch.LongTensor{chooseInt(1, sz2), chooseInt(1, sz2)}
   src = torch.FloatTensor(sz1, 2):uniform():cuda()
   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local x, src = x:type(ctype), src:type(ctype)
      compareCPUAndCUDATypeTensorArgs(typename, true, x, 'indexCopy',
                                      index, longIndex, src)
      if typename ~= 'torch.CudaByteTensor' and typename ~= 'torch.CudaCharTensor' then
          compareCPUAndCUDATypeTensorArgs(typename, false, x, 'indexCopy',
                                          index, longIndex, src)
      end
   end

   -- Case 3: 1D tensor, indexCopy over 1st dimension, 2 indices
   x = torch.FloatTensor():rand(sz1)
   index = 1
   longIndex = torch.LongTensor{chooseInt(1, sz1), chooseInt(1, sz1)}
   src = torch.FloatTensor(2):uniform()
   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local x, src = x:type(ctype), src:type(ctype)
      compareCPUAndCUDATypeTensorArgs(typename, true, x, 'indexCopy',
                                      index, longIndex, src)
      if typename ~= 'torch.CudaByteTensor' and typename ~= 'torch.CudaCharTensor' then
          compareCPUAndCUDATypeTensorArgs(typename, false, x, 'indexCopy',
                                          index, longIndex, src)
      end
   end

   tester:assert(isEqual(
      x:cuda():indexCopy(index, longIndex:cuda(), src:cuda()),
      x:indexCopy(index, longIndex, src)),
      "Divergent results between CPU and CUDA for function 'indexCopy'")

   checkMultiDevice(x, 'indexCopy', index, longIndex, src)
end

local function testIndexAdd(types, gpu2cpu_map)
   local sz1 = chooseInt(minsize, maxsize) -- dim1
   local sz2 = chooseInt(minsize, maxsize) -- dim2
   local x = torch.FloatTensor():rand(sz1, sz2) -- input

   -- Case 1: 2D tensor, indexAdd over first dimension, 2 indices
   -- choose two indices from the first dimension, i.e. [1,sz1]
   local longIndex = torch.LongTensor{chooseInt(1, sz1), chooseInt(1, sz1)}
   local index = 1
   local src = torch.FloatTensor(2, sz2):uniform()

   for k, typename in ipairs(types) do
      local ctype = t2cpu[typename]
      local x, src = x:type(ctype), src:type(ctype)
      compareCPUAndCUDATypeTensorArgsWithConvInternal(typename, gpu2cpu_map, true, nil, x, 'indexAdd',
                                              index, longIndex, src)
      if typename ~= 'torch.CudaByteTensor' and typename ~= 'torch.CudaCharTensor' then
          compareCPUAndCUDATypeTensorArgsWithConvInternal(typename, gpu2cpu_map, false, nil, x, 'indexAdd',
                                                  index, longIndex, src)
      end
   end

   -- Case 2: 2D tensor, indexAdd over second dimension, 2 indices
   index = 2
   longIndex =  torch.LongTensor{chooseInt(1, sz2), chooseInt(1, sz2)}
   src = torch.FloatTensor(sz1, 2):uniform():cuda()
   for k, typename in ipairs(types) do
      local ctype = t2cpu[typename]
      local x, src = x:type(ctype), src:type(ctype)
      compareCPUAndCUDATypeTensorArgsWithConvInternal(typename, gpu2cpu_map, true, nil, x, 'indexAdd',
                                              index, longIndex, src)
      if typename ~= 'torch.CudaByteTensor' and typename ~= 'torch.CudaCharTensor' then
          compareCPUAndCUDATypeTensorArgsWithConvInternal(typename, gpu2cpu_map, false, nil, x, 'indexAdd',
                                                  index, longIndex, src)
      end
   end

   -- Case 3: 1D tensor, indexAdd over 1st dimension, 2 indices
   x = torch.FloatTensor():rand(sz1)
   index = 1
   longIndex = torch.LongTensor{chooseInt(1, sz1), chooseInt(1, sz1)}
   src = torch.FloatTensor(2):uniform()
   for k, typename in ipairs(types) do
      local ctype = t2cpu[typename]
      local x, src = x:type(ctype), src:type(ctype)
      compareCPUAndCUDATypeTensorArgsWithConvInternal(typename, gpu2cpu_map, true, nil, x, 'indexAdd',
                                              index, longIndex, src)
      if typename ~= 'torch.CudaByteTensor' and typename ~= 'torch.CudaCharTensor' then
          compareCPUAndCUDATypeTensorArgsWithConvInternal(typename, gpu2cpu_map, false, nil, x, 'indexAdd',
                                                  index, longIndex, src)
      end
   end

   tester:assert(isEqual(
      x:cuda():indexAdd(index, longIndex:cuda(), src:cuda()),
      x:indexAdd(index, longIndex, src)),
      "Divergent results between CPU and CUDA for function 'indexAdd'")

   checkMultiDevice(x, 'indexAdd', index, longIndex, src)
end

function test.indexAdd()
   testIndexAdd(typenames)
end

function test.indexAddHalf()
   -- don't have cpu versions of half, so let's compare with float.
   -- additional divergence due to float/half:
   -- half_digits_precision = log10(2^11) ~ 3, reserve another
   -- digit to be safe
   if cutorch.hasHalf then
      local old_tolerance = test_tolerance
      test_tolerance = test_tolerance + 1e-2;
      local halfOnly = { 'torch.CudaHalfTensor' }
      local halft2gpu2 = {
        ['torch.FloatTensor'] = 'torch.CudaHalfTensor',
        ['torch.LongTensor'] = 'torch.CudaLongTensor'
      }
      testIndexAdd(halfOnly, halft2gpu2)
      local test_tolerance =  old_tolerance
  end
end

function test.indexFill()
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)

   local longIndex = torch.LongTensor{chooseInt(1, sz1), chooseInt(1, sz1)}
   local index = 1
   local val = torch.random(10)
   for k, typename in ipairs(typenames) do
       local x = x:type(t2cpu[typename])
       compareCPUAndCUDATypeTensorArgs(typename, true, x, 'indexFill',
                                       index, longIndex, val)
       if typename ~= 'torch.CudaByteTensor' and typename ~= 'torch.CudaCharTensor' then
           compareCPUAndCUDATypeTensorArgs(typename, false, x, 'indexFill',
                                           index, longIndex, val)
       end
   end
   index = 2
   longIndex =  torch.LongTensor{chooseInt(1, sz2), chooseInt(1, sz2)}
   val = torch.random(10)
   for k, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, true, x, 'indexFill',
                                      index, longIndex, val)
      if typename ~= 'torch.CudaByteTensor' and typename ~= 'torch.CudaCharTensor' then
          compareCPUAndCUDATypeTensorArgs(typename, false, x, 'indexFill',
                                          index, longIndex, val)
      end
   end

   x = torch.FloatTensor():rand(sz1)
   index = 1
   longIndex = torch.LongTensor{chooseInt(1, sz1), chooseInt(1, sz1)}
   val = torch.random(10)
   for k, typename in ipairs(typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, true, x, 'indexFill',
                                      index, longIndex, val)
      if typename ~= 'torch.CudaByteTensor' and typename ~= 'torch.CudaCharTensor' then
          compareCPUAndCUDATypeTensorArgs(typename, false, x, 'indexFill',
                                          index, longIndex, val)
      end
   end

   tester:assert(isEqual(
      x:cuda():indexFill(index, longIndex:cuda(), val),
      x:indexFill(index, longIndex, val)),
      "Divergent results between CPU and CUDA for function 'indexFill'")

   checkMultiDevice(x, 'indexFill', index, longIndex, val)
end

function test.norm()
   for n = 0, 3 do
     local cpu = torch.FloatTensor(chooseInt(20, 50), 2):uniform(-0.5, 0.5)
     for _, typename in ipairs(float_typenames) do
        local x = cpu:type(t2cpu[typename])
        compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'norm', n)
        compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'norm', n, 1)
        compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'norm', n, 2)
     end
   end

   for i = 1, 5 do
      for n = 0, 3 do
         local cpu = torch.FloatTensor(chooseInt(20, 50), 2):uniform(-0.5, 0.5)

         if torch.random(1, 2) == 1 then
            cpu = cpu:transpose(1, 2)
         end

         compareFloatAndCuda(cpu, 'norm', n)
         compareFloatAndCuda(cpu, 'norm', n, 1)
         compareFloatAndCuda(cpu, 'norm', n, 2)
      end
   end
end

function test.renorm()
   local x = torch.randn(10,5):float()
   local maxnorm = x:norm(2,1):mean()

   for _, typename in ipairs(float_typenames) do
      local x = x:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'renorm', 2, 2, maxnorm)
   end

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

function test.dist()
   local minsize = 5
   local maxsize = 10
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local x = torch.FloatTensor():rand(sz1, sz2)
   local y = torch.FloatTensor():rand(sz1, sz2)
   for _, typename in ipairs(float_typenames) do
       local x = x:type(t2cpu[typename])
       local y = y:type(t2cpu[typename])
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'dist', y)
       compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'dist', y, 3)
   end
   checkMultiDevice(x, 'dist', y)
end

function test.indexCopy2()
   for tries = 1, 5 do
      local t = createTestTensor(1000000)
      local selectdim = chooseInt(1, t:nDimension())
      local indices = torch.randperm(t:size(selectdim)):long()

      compareFloatAndCudaTensorArgs(
          t, 'indexCopy', selectdim, indices, t:clone())
   end
end

function test.indexAdd2()
   for tries = 1, 5 do
      local t = createTestTensor(1000000)
      local selectdim = chooseInt(1, t:nDimension())
      local indices = torch.randperm(t:size(selectdim)):long()

      compareFloatAndCudaTensorArgs(
          t, 'indexAdd', selectdim, indices, t:clone())
   end
end

function test.indexFill2()
   for tries = 1, 5 do
      local t = createTestTensor(1000000)
      local selectdim = chooseInt(1, t:nDimension())
      local numIndices = chooseInt(1, t:size(selectdim))
      local indices = torch.randperm(numIndices):long()

      compareFloatAndCuda(t, 'indexFill', selectdim, indices, 1)
   end
end

function test.indexSelect2()
   for tries = 1, 5 do
      local t = createTestTensor(1000000)
      local selectdim = chooseInt(1, t:nDimension())
      local numIndices = chooseInt(1, t:size(selectdim))
      local indices = torch.randperm(numIndices):long()

      compareFloatAndCuda(t, 'index', selectdim, indices)
   end
end

function test.cross()
   -- Test finding the first non-zero dimension
   local x = torch.FloatTensor():randn(4,3,2,3)
   local y = torch.FloatTensor():randn(4,3,2,3)
   compareFloatAndCudaTensorArgs(x, 'cross', y)
   checkMultiDevice(x, 'cross', y)

   for tries = 1, 5 do
      local nelems = 10000000
      local ndims = chooseInt(1, 10)
      local crossdim = chooseInt(1, ndims)
      sizes = {}
      for i = 1, ndims do
         sizes[i] = chooseInt(1, math.min(20, math.sqrt(nelems)))
         nelems = nelems / sizes[i]
      end
      sizes[crossdim] = 3
      local x = torch.FloatTensor():randn(unpack(sizes))
      local y = torch.FloatTensor():randn(unpack(sizes))
      for _, typename in ipairs(typenames) do
         local x = x:type(t2cpu[typename])
         local y = y:type(t2cpu[typename])
         compareCPUAndCUDATypeTensorArgs(typename, nil, x, 'cross', y, crossdim)
         checkMultiDevice(x, 'cross', y, crossdim)
      end
   end
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

function test.addbmm()
    local sizes = {
        {16, 3, 1, 4},
        {1, 12, 1, 7},
        {24, 23, 22, 21},
        {1, 1, 1, 1},
        {1, 1, 7, 4},
        {12, 1, 12, 1},
        {10, 10, 10, 10},
    }
    local old_tt = test_tolerance
    test_tolerance = 1e-3
    local multiCheck = false
    for _, size in pairs(sizes) do
        local b, n, k, m = unpack(size)
        local cs = torch.randn(n, m)
        local as = torch.randn(b, n, k)
        local bs = torch.randn(b, k, m)
        local beta = torch.randn(1)[1]
        local alpha = torch.randn(1)[1]
        compareFloatAndCudaTensorArgs(cs, 'addbmm', beta, cs, alpha, as, bs)
        if not multiCheck then -- just check multidevice once
            checkMultiDevice(cs, 'addbmm', as, bs)
            multiCheck = true
        end
    end
    test_tolerance = old_tt
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

function test.inverse()
   local a = torch.eye(5):add(torch.Tensor(5, 5):uniform(-0.1, 0.1))
   for _, typename in ipairs({'torch.DoubleTensor', 'torch.FloatTensor'}) do
       local at = a:type(typename)
       local i1 = torch.inverse(at)
       local i2 = torch.inverse(a:cuda())
       tester:assertle((i2 - i1:cuda()):abs():max(), 1e-5, "wrong inverse answer")
   end
end

if cutorch.magma then
   function test.gesv()
      local a = torch.Tensor(5, 5):uniform(-1, 1)
      local b = torch.Tensor(5, 3):uniform(-1, 1)
      for _, typename in ipairs({'torch.DoubleTensor', 'torch.FloatTensor'}) do
          local at = a:type(typename)
          local bt = b:type(typename)
          local rb1, ra1 = torch.gesv(bt, at)
          local rb2, ra2 = torch.gesv(bt:cuda(), at:cuda())
          tester:assertle((rb2 - rb1:cuda()):abs():max(), 1e-5, "wrong gesv answer")
          tester:assertle((ra2 - ra1:cuda()):abs():max(), 1e-5, "wrong gesv answer")
      end
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
      for _, typename in ipairs({'torch.DoubleTensor', 'torch.FloatTensor'}) do
          local at = a:type(typename)
          local bt = b:type(typename)
          local rb1, ra1 = torch.gels(bt, at)
          local rb2, ra2 = torch.gels(bt:cuda(), at:cuda())
          tester:assertle((rb2 - rb1:cuda()):abs():max(), 5e-4, "wrong gels answer")
          tester:assertle((ra2 - ra1:cuda()):abs():max(), 5e-4, "wrong gels answer")
      end
   end

   function test.symeig()
      local a = torch.Tensor({{ 1.96,  0.00,  0.00,  0.00,  0.00},
                              {-6.49,  3.80,  0.00,  0.00,  0.00},
                              {-0.47, -6.39,  4.17,  0.00,  0.00},
                              {-7.20,  1.50, -1.51,  5.70,  0.00},
                              {-0.65, -6.34,  2.67,  1.80, -7.10}}):t()
      for _, typename in ipairs({'torch.DoubleTensor', 'torch.FloatTensor'}) do
          local at = a:type(typename)
          local e1,v1 = torch.symeig(at, 'V')
          local e2,v2 = torch.symeig(at:cuda(), 'V')
          tester:assertle((e2 - e1:cuda()):abs():max(), 1e-5, "wrong symeig answer")
          tester:assertle((v2 - v1:cuda()):abs():max(), 1e-5, "wrong symeig answer")
      end
   end

   function test.eig()
      local a = torch.Tensor{
         {-0.1425, -0.4750, -0.8551, 0.6729, -0.7453},
         {-0.2696,  0.4330,  0.5077, 0.3709, -0.6053},
         { 0.4330,  0.6727, -0.5049, 0.4600,  0.6249},
         { 0.5766, -0.6743,  0.6903, 0.3646, -0.4571},
         {-0.8956, -0.4074, -0.7583, 0.1838, -0.0091},
      }
      for _, typename in ipairs({'torch.DoubleTensor', 'torch.FloatTensor'}) do
          local at = a:type(typename)
          local e1,v1 = torch.eig(at, 'V')
          local e2,v2 = torch.eig(at:cuda(), 'V')
          tester:assertle((e2 - e1:cuda()):abs():max(), 1e-6, "wrong eig answer")
          tester:assertle((v2:abs() - v1:abs():cuda()):abs():max(), 1e-6, "wrong eig answer")
      end
   end

   function test.svd()
      local a = torch.CudaTensor{
         {8.79,  6.11, -9.15,  9.57, -3.49,  9.84},
         {9.93,  6.91, -7.93,  1.64,  4.02,  0.15},
         {9.83,  5.04,  4.86,  8.83,  9.80, -8.99},
         {5.45, -0.27,  4.85,  0.74, 10.00, -6.02},
         {3.16,  7.98,  3.01,  5.80,  4.27, -5.31}}

      for _, typename in ipairs({'torch.CudaDoubleTensor', 'torch.CudaTensor'}) do
          local at = a:type(typename)
          local u,s,v = torch.svd(a, 'A')

          local temp = torch.Tensor(a:size(2)):zero()
          temp:narrow(1, 1, a:size(1)):copy(s)
          local sigma = torch.diag(temp):resize(a:size(1), a:size(2)):cuda()

          local m = u * sigma * v:t()

          tester:assertle((m - a):abs():max(), 1e-5, "svd: a != u * s * vT")
          tester:assertle((u*u:t() - torch.eye(a:size(1)):cuda()):abs():max(), 1e-6, "svd: u should be unitary")
          tester:assertle((v*v:t() - torch.eye(a:size(2)):cuda()):abs():max(), 1e-6, "svd: v should be unitary")
      end
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

      for _, typename in ipairs({'torch.DoubleTensor', 'torch.FloatTensor'}) do
          local at = A:type(typename)
          for _, triarg in ipairs({'U', 'L'}) do
              local chol  = torch.potrf(at, triarg)

              local i1 = torch.potri(chol, triarg)
              local i2 = torch.potri(chol:cuda(), triarg)
              local M = at:cuda() * i2
              tester:assertle((i2 - i1:cuda()):abs():max(), 1e-5, "wrong potri answer")
              tester:assertle((M - torch.eye(at:size(1)):cuda()):abs():max(), 1e-5, "potri not an inverse")
          end
      end
   end

   function test.potrf()
      local A = torch.Tensor{
         { 8.7937, 0.5104, 1.5955,-0.6738,-3.3883},
         { 0.5104, 1.4286, 0.0236, 0.4734, 0.2807},
         { 1.5955, 0.0236, 1.4539,-1.1123, 0.8161},
         {-0.6738, 0.4734,-1.1123, 2.4071,-1.2756},
         {-3.3883, 0.2807, 0.8161,-1.2756, 4.3415},
      }
      for _, typename in ipairs({'torch.DoubleTensor', 'torch.FloatTensor'}) do
          local at = A:type(typename)
          for _, triarg in ipairs({'U', 'L'}) do
              local i1 = torch.potrf(at, triarg)
              local i2 = torch.potrf(at:cuda(), triarg)
              tester:assertle((i2 - i1:cuda()):abs():max(), 1e-5, "wrong potrf answer")
          end
      end
   end

   function test.potrs()
      local A = torch.Tensor({
        {1.2705,  0.9971,  0.4948,  0.1389,  0.2381},
        {0.9971,  0.9966,  0.6752,  0.0686,  0.1196},
        {0.4948,  0.6752,  1.1434,  0.0314,  0.0582},
        {0.1389,  0.0686,  0.0314,  0.0270,  0.0526},
        {0.2381,  0.1196,  0.0582,  0.0526,  0.3957}})
      local B = torch.Tensor({
        {0.6219,  0.3439,  0.0431},
        {0.5642,  0.1756,  0.0153},
        {0.2334,  0.8594,  0.4103},
        {0.7556,  0.1966,  0.9637},
        {0.1420,  0.7185,  0.7476}})
      for _, typename in ipairs({'torch.DoubleTensor', 'torch.FloatTensor'}) do
          local at = A:type(typename)
          local bt = B:type(typename)
          for _, triarg in ipairs({'U', 'L'}) do
              local chol = torch.potrf(at, triarg)
              local solve1 = torch.potrs(bt, chol, triarg)
              local solve2 = torch.potrs(bt:cuda(), chol:cuda(), triarg)
              tester:assertle((solve2 - solve1:cuda()):abs():max(), 1e-4, "wrong potrs answer")
          end
      end
   end

   function test.qr()
      local A = torch.Tensor{
         { 0.9023,  1.5967,  0.3388, -0.0746, -0.5717},
         {-2.0442,  2.3974, -1.0883,  0.4018, -0.3938},
         {-0.1065, -1.3180,  0.3542,  1.3684,  0.3934},
         {-0.2987,  1.9035, -1.4192, -0.9738,  1.4384},
         {-0.5315,  0.4958,  0.4449, -0.4676, -0.4878},
      }
      for _, typename in ipairs({'torch.DoubleTensor', 'torch.FloatTensor'}) do
          local at = A:type(typename)
          local q1,r1 = torch.qr(at)
          local q2,r2 = torch.qr(at:cuda())
          tester:assertle((q2 - q1:cuda()):abs():max(), 1e-5, "wrong qr answer")
          tester:assertle((r2 - r1:cuda()):abs():max(), 1e-5, "wrong qr answer")
      end
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

function test.isSetTo()
  local t1 = torch.CudaTensor(7, 4, 9)
  local t2 = torch.CudaTensor(7, 8, 2)
  local t3 = t2:view(7*8*2)
  tester:assert(t1:isSetTo(t2) == false, "t1 and t2 are not the same tensor. ")
  tester:assert(t2:isSetTo(t3) == false, "t2 and t3 share storage but are different views. ")
  t2:set(t1)
  tester:assert(t1:isSetTo(t2) == true, "t1 and t2 are the same tensor now.")
  tester:assert(t2:isSetTo(t1) == true, "by symmetry. ")
  tester:assert(t3:isSetTo(t1) == false, "now they are completely unrelated.")
end

function test.isSize()
   local t1 = torch.CudaTensor(3, 4, 5)
   local s1 = torch.LongStorage({3, 4, 5})
   local s2 = torch.LongStorage({5, 4, 3})

   tester:assert(t1:isSize(s1) == true, "wrong answer ")
   tester:assert(t1:isSize(s2) == false, "wrong answer ")
   tester:assert(t1:isSize(t1:size()) == true, "wrong answer ")
end

function test.elementSize()
  local float = torch.CudaStorage():elementSize()
  tester:asserteq(float, torch.CudaTensor():elementSize())
  tester:assertne(float, 0)
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

   for _, typename in ipairs(float_typenames) do
       local x = t:type(typename)
       x:uniform(min, max)
       checkIfUniformlyDistributed(x, min, max)
   end
   checkMultiDevice(t, 'uniform', min, max)
end

function test.bernoulli()
   local minsize = 1000
   local maxsize = 2000
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local p = torch.uniform()
   local p_fl = torch.rand(sz1, sz2):cuda()
   local p_dbl = torch.rand(sz1, sz2):cudaDouble()
   local t = torch.CudaTensor(sz1, sz2)

   for _, typename in ipairs(typenames) do
       local x = t:type(typename)
       local expected_mean
       for i, p in ipairs({p, p_fl, p_dbl}) do
          x:bernoulli(p)
          local mean = x:sum() / (sz1 * sz2)
          if torch.type(p) == 'number' then
             expected_mean = p
          else
             expected_mean = p:mean()
          end
          tester:assertalmosteq(mean, expected_mean, 0.1, "mean is not equal to the expected value")
          local f = x:float()
          tester:assertTensorEq(f:eq(1):add(f:eq(0)):float(),
                                torch.FloatTensor(sz1, sz2):fill(1),
                                1e-6,
                                "each value must be either 0 or 1")
       end
   end
   checkMultiDevice(t, 'bernoulli', p)
end

function test.normal()
   local minsize = 1000
   local maxsize = 2000
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local mean, std = torch.uniform(), 0.1 * torch.uniform()
   local tolerance = 0.01
   local t = torch.CudaTensor(sz1, sz2)

   for _, typename in ipairs(float_typenames) do
       local x = t:type(t2cpu[typename])
       x:normal(mean, std)
       tester:assertalmosteq(x:mean(), mean, tolerance, "mean is wrong")
       tester:assertalmosteq(x:std(), std, tolerance, "standard deviation is wrong")
   end

   checkMultiDevice(t, 'normal', mean, std)
end

function test.logNormal()
   local minsize = 1000
   local maxsize = 2000
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local mean, std = torch.uniform(), 0.1 * torch.uniform()
   local tolerance = 0.01
   local t = torch.CudaTensor(sz1, sz2)

   for _, typename in ipairs(float_typenames) do
       local x = t:type(typename)
       x:logNormal(mean, std)
       local logt = x:log()
       tester:assertalmosteq(logt:mean(), mean, tolerance, "mean is wrong")
       tester:assertalmosteq(logt:std(), std, tolerance, "standard deviation is wrong")
   end
   checkMultiDevice(t, 'logNormal', mean, std)
end

function test.geometric()
   local minsize = 1000
   local maxsize = 2000
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)

   -- unlike other tests, we pick a large p-value to lower the variance, so
   -- that its highly unlikely the mean falls outside the bounds of the
   -- specified tolerance
   local p = 0.8
   local tolerance = 0.2

   local t = torch.CudaTensor(sz1, sz2)
   local mean = (1 / p)

   for _, typename in ipairs(float_typenames) do
       local x = t:type(typename)
       x:geometric(p)
       tester:assertalmosteq(x:mean(), mean, tolerance, "mean is wrong")
   end
   checkMultiDevice(t, 'geometric', p)
end

function test.exponential()
   local minsize = 1000
   local maxsize = 2000
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local lambda = torch.uniform()
   local t = torch.CudaTensor(sz1, sz2)

   for _, typename in ipairs(float_typenames) do
       local x = t:type(t2cpu[typename])
       x:exponential(lambda)
       local u = torch.FloatTensor(sz1, sz2):fill(1) -
                     (x:float() * -lambda):exp()
       checkIfUniformlyDistributed(u, 0, 1)
   end
   checkMultiDevice(t, 'exponential', lambda)
end

function test.cauchy()
   local minsize = 1000
   local maxsize = 2000
   local sz1 = chooseInt(minsize, maxsize)
   local sz2 = chooseInt(minsize, maxsize)
   local median, sigma = torch.uniform(), torch.uniform()
   local t = torch.CudaTensor(sz1, sz2)

   for _, typename in ipairs(float_typenames) do
       local x = t:type(typename)
       x:cauchy(median, sigma)
       local u = ((x:float() - median) / sigma):atan() / math.pi + 0.5
       checkIfUniformlyDistributed(u, 0, 1)
   end
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
      for _, typename in ipairs(float_typenames) do
          if typename ~= 'torch.CudaHalfTensor' then
             local pd = prob_dist:type(typename)
             local sample_indices = torch.multinomial(pd, n_sample, true)
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
      for _, typename in ipairs(float_typenames) do
          if typename ~= 'torch.CudaHalfTensor' then
             local pd = prob_dist:type(typename)
             local sample_indices = torch.multinomial(pd, n_sample, false)
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

      local orig = t:cudaLong()

      for _, typename in ipairs(float_typenames) do
          -- Half tensors have precision errors for the binary search causing this test
          -- to fail frequently
          if typename ~= 'torch.CudaHalfTensor' then
              local x = t:type(typename)

              -- Sample without replacement
              local result = torch.multinomial(x, distSize)
              tester:assert(result:size(1) == distributions)
              tester:assert(result:size(2) == distSize)

              -- Sort, and we should have the original results, since without replacement
              -- sampling everything, we should have chosen every value uniquely
              result = result:sort(2)
              tester:assertTensorEq(orig, result, 0, "error in multinomial_without_replacement_gets_all")
          end
      end
   end
end

function test.multinomial_vector()
   local n_col = torch.random(100)
   local prob_dist = torch.CudaTensor(n_col):uniform()
   local n_sample = n_col
   for _, typename in ipairs(float_typenames) do
       if typename ~= 'torch.CudaHalfTensor' then
           local pd = prob_dist:type(typename)
           local sample_indices = torch.multinomial(pd, n_sample, true)
           tester:assert(sample_indices:dim() == 1, "wrong sample_indices dim")
           -- Multinomial resizes prob_dist to be 2d (1xn), check that the resize
           -- was undone
           tester:assert(prob_dist:dim() == 1, "wrong number of prob_dist dimensions")
           tester:assert(sample_indices:size(1) == n_sample, "wrong number of samples")
       end
   end
end

function test.multinomial_alias()
   for tries = 1, 10 do
      local n_class = torch.random(100)
      local prob_dist = torch.CudaTensor(n_class):uniform()
      local n_sample = torch.random(100)
      local dim_1 = torch.random(10)
      for _, typename in ipairs(float_typenames) do
	 if typename ~= 'torch.CudaHalfTensor' then
	    -- Get the probability distribution
             local pd = prob_dist:type(typename)
	     local state = torch.multinomialAliasSetup(pd)

	     -- Checking the validity of the setup tables
	     tester:assert(state[1]:min() >= 0, "alias indices has an index below or equal to 0(cold)")
	     tester:assert(state[1]:max() < n_class, state[1]:max().." alias indices has an index exceeding num_class(cold)")

	     --Checking the same things if the memory is already allocated
	     local state = torch.multinomialAliasSetup(pd, state)
	     tester:assert(state[1]:min() >= 0, "alias indices has an index below or equal to 0(hot)")
	     tester:assert(state[1]:max() < n_class, state[1]:max().." alias indices has an index exceeding num_class(hot)")

	     --Generating a 1d and a 2d long tensor to be filled with indices
	     local sample_indices = torch.CudaLongTensor(n_sample)
	     local sample_indices_dim2 = torch.CudaLongTensor(n_sample/dim_1, dim_1)
	     local state = {state[1], state[2]:type('torch.CudaTensor')}
	     cutorch.synchronize()
	     torch.multinomialAlias(sample_indices, state)
	     cutorch.synchronize()
	     torch.multinomialAlias(sample_indices_dim2:view(-1), state)

	     --Checking the validity of the sampled indices
             tester:assert(sample_indices_dim2:dim() == 2, "wrong sample_indices dim")
             tester:assert(sample_indices_dim2:size(2) == dim_1, "wrong number of samples")
	     tester:assert(sample_indices:min() > 0, sample_indices:min().."sampled indices has an index below or equal to 0")
	     tester:assert(sample_indices:max() <= n_class, sample_indices:max().."indices has an index exceeding num_class")
	     tester:assert(sample_indices_dim2:min() > 0, sample_indices_dim2:min().."sampled indices has an index below or equal to 0")
	     tester:assert(sample_indices_dim2:max() <= n_class, sample_indices_dim2:max().."indices has an index exceeding num_class")
   
         end
      end
   end
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
       tester:assert(tensors[i]:getDevice() == tensors[i]:storage():getDevice(),
          "tensor's device id doesn't match its storage's device id")
    end
    cutorch.setDevice(1) -- reset device
end

function test.multi_gpu_copy_noncontig()
   local srcDevice = 1
   local dstDevice = cutorch.getDeviceCount()

   local t1, t2
   for transposeSrc = 0,1 do
     for transposeDst = 0,1 do
        cutorch.withDevice(
           srcDevice,
           function()
              t1 = torch.CudaTensor(100000, 1000):fill(1)
              cutorch.synchronize()
        end)

        cutorch.withDevice(
           dstDevice,
           function()
              t2 = torch.CudaTensor(100000, 1000):fill(2)
              cutorch.synchronize()
        end)

        if transposeSrc == 1 then -- maybe make t1 non-contiguous
           cutorch.withDevice(srcDevice, function() t1=t1:transpose(1,2) end)
        end
        if transposeDst == 1 then -- maybe make t2 non-contiguous
           cutorch.withDevice(dstDevice, function() t2=t2:transpose(1,2) end)
        end

        -- try to induce a race on t2
        cutorch.withDevice(dstDevice, function() t2:fill(3) end)

        -- perform the copy
        -- CudaTensor:copy() should not depend on the current device
        t2:copy(t1)

        -- try to induce a race on t1
        cutorch.withDevice(srcDevice, function() t1:fill(4) end)

        local t2_max
        cutorch.withDevice(dstDevice, function() t2_max = t2:max() end)
        tester:assert(t2_max == 1, "bad copy, transposeSrc= " .. transposeSrc ..
               " transposeDst= " .. transposeDst .. ". t2:max() = " .. t2_max)
      end
   end
end

function test.cudaTypeCopy()

   local types = {
      {'float', 'FloatTensor'},
      {'byte',  'ByteTensor'},
      {'char',  'CharTensor'},
      {'short', 'ShortTensor'},
      {'int',   'IntTensor'},
      {'long',  'LongTensor'},
      {'double','DoubleTensor'},
      {'half', 'HalfTensor'},
      {'cuda',      'CudaTensor'},
      {'cudaByte',  'CudaByteTensor'},
      {'cudaChar',  'CudaCharTensor'},
      {'cudaShort', 'CudaShortTensor'},
      {'cudaInt',   'CudaIntTensor'},
      {'cudaLong',  'CudaLongTensor'},
      {'cudaDouble','CudaDoubleTensor'},
   }
   if cutorch.hasHalf then
      table.insert(types, {'cudaHalf', 'CudaHalfTensor'})
   end

   local N = 100
   local t0 = torch.range(1,12):reshape(3,4)

   -- t carries over from one iteration to the next
   local t = t0:clone()
   for i = 1, N do
      -- convert to a random (CPU or GPU) type)
      local conversionFunc, tensorSubtype = unpack(types[torch.random(#types)])
      local tensorType = 'torch.' .. tensorSubtype

      if torch.random(0,1) ~= 0 then
         -- this is equivalent to t = t:float()
         t = t[conversionFunc](t)
      else
         -- this is equivalent to t = torch.XTensor():copy(t)
         t = torch[tensorSubtype](3,4):copy(t)
      end

      -- check the type
      tester:assert(t:type() == tensorType, t:type() .. ' ~= ' .. tensorType)

      -- check metadata
      tester:assert(t:isContiguous())
      tester:assert(t:size(1) == 3 and t:size(2) == 4)
      tester:assert(t:nDimension() == 2)

      -- check data
      tester:assertTensorEq(t:double(), t0, 0)


      -- check indexing
      -- FIXME: doesn't work yet
      -- tester:assert(ct[{1,1}] == 1)
   end

   -- check narrowing conversions
   tester:assert(torch.Tensor(1):fill(500):cudaByte():float()[1] == 244)
   tester:assert(torch.Tensor(1):fill(500):cudaChar():float()[1] == -12)
end


function test.cudaStorageTypeCopy()

   local types = {
      {'float', 'FloatStorage'},
      {'byte',  'ByteStorage'},
      {'char',  'CharStorage'},
      {'short', 'ShortStorage'},
      {'int',   'IntStorage'},
      {'long',  'LongStorage'},
      {'double','DoubleStorage'},
      {'half',   'HalfStorage'},
      {'cuda',      'CudaStorage'},
      {'cudaByte',  'CudaByteStorage'},
      {'cudaChar',  'CudaCharStorage'},
      {'cudaShort', 'CudaShortStorage'},
      {'cudaInt',   'CudaIntStorage'},
      {'cudaLong',  'CudaLongStorage'},
      {'cudaDouble','CudaDoubleStorage'},
   }
   if cutorch.hasHalf then
      table.insert(types, {'cudaHalf', 'CudaHalfStorage'})
   end

   local N = 100
   local t0 = torch.range(1,12):reshape(3,4):storage()

   -- t carries over from one iteration to the next
   local t = torch.DoubleStorage(t0:size()):copy(t0)
   for i = 1, N do
      -- convert to a random (CPU or GPU) type)
      local conversionFunc, storageSubtype = unpack(types[torch.random(#types)])
      local storageType = 'torch.' .. storageSubtype

      -- this is equivalent to t = torch.XStorage():copy(t)
      t = torch[storageSubtype](12):copy(t)

      -- check the type
      tester:assert(torch.type(t) == storageType, torch.type(t) .. ' ~= ' .. storageType)

      local d = torch.DoubleStorage(12):copy(t)
      for i = 1, t:size() do
         tester:assert(d[i] == t0[i], storageSubtype .. ': ' .. i .. ': ' .. d[i] .. ' ~= ' .. t0[i])
      end
   end
end

function test.tensorToTable()
   local types = {
      {'CudaTensor',       'FloatTensor'},
      {'CudaByteTensor',   'ByteTensor'},
      {'CudaCharTensor',   'CharTensor'},
      {'CudaShortTensor',  'ShortTensor'},
      {'CudaIntTensor',    'IntTensor'},
      {'CudaLongTensor',   'LongTensor'},
      {'CudaDoubleTensor', 'DoubleTensor'},
   }
   if cutorch.hasHalf then
      table.insert(types, {'CudaHalfTensor', 'HalfTensor'})
   end
   for _, types in ipairs(types) do
      local cudaType, hostType = unpack(types)
      local dim = torch.random(5)
      local size = torch.LongTensor(dim):random(5):totable()
      local hostTensor = nil
      if hostType ~= 'HalfTensor' then
          hostTensor = torch[hostType](size):random()
      else
          -- work around HalfTensor not having random functions and reduced range
          local copyTensor = torch['FloatTensor'](size):random(128)
          hostTensor = torch[hostType](size)
          hostTensor:copy(copyTensor)
      end
      local cudaTensor = torch[cudaType](size):copy(hostTensor)
      tester:assertTableEq(hostTensor:totable(), cudaTensor:totable(),
                           'wrong result for ' .. cudaType .. ':totable()')
   end
end

function test.storageToTable()
   local types = {
      {'CudaStorage',       'FloatTensor'},
      {'CudaByteStorage',   'ByteTensor'},
      {'CudaCharStorage',   'CharTensor'},
      {'CudaShortStorage',  'ShortTensor'},
      {'CudaIntStorage',    'IntTensor'},
      {'CudaLongStorage',   'LongTensor'},
      {'CudaDoubleStorage', 'DoubleTensor'},
   }
   if cutorch.hasHalf then
     types['CudaHalfStorage'] = 'HalfTensor'
   end

   for _, types in ipairs(types) do
      local cudaStorageType, hostTensorType = unpack(types)
      local size = torch.random(10)
      hostTensor = torch[hostTensorType](size):random()
      cudaStorage = torch[cudaStorageType](size):copy(hostTensor:storage())
      tester:assertTableEq(hostTensor:storage():totable(), cudaStorage:totable(),
                           'wrong result for ' .. cudaStorageType .. ':totable()')
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
   mask=mask:cudaByte()
   local y_cuda = x:maskedSelect(mask)
   tester:assertTensorEq(y, y_cuda:float(), 0.00001, "Error in maskedSelect")
   checkMultiDevice(x, 'maskedSelect', mask)

   -- non-contiguous, no result tensor, cuda mask
   local x = torch.randn(n_row, n_col):float()
   local mask = torch.ByteTensor(n_row,n_col):bernoulli()
   local y = x:t():maskedSelect(mask)
   x=x:cuda()
   mask=mask:cudaByte()
   local y_cuda = x:t():maskedSelect(mask)
   tester:assertTensorEq(y, y_cuda:float(), 0.00001,
                         "Error in maskedSelect non-contiguous")

   -- contiguous, with result tensor, cuda mask
   local x = torch.randn(n_row, n_col):float()
   local mask = torch.ByteTensor(n_row,n_col):bernoulli()
   local y = torch.FloatTensor()
   y:maskedSelect(x, mask)
   x=x:cuda()
   mask=mask:cudaByte()
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
   mask=mask:cudaByte()
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
   mask=mask:cudaByte()
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
   mask=mask:cudaByte()
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
   mask=mask:cudaByte()
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
   mask=mask:cudaByte()
   x_cuda:maskedFill(mask, 334)
   tester:assertTensorEq(x, x_cuda:float(), 0.00001, "Error in maskedFill")
   checkMultiDevice(x_cuda, 'maskedFill', mask, 334)

   -- non-contiguous, no result tensor, cuda mask
   local x = gt:clone()
   mask = mask:byte()
   x:t():maskedFill(mask, 334)
   local x_cuda = gt:cuda()
   mask=mask:cudaByte()
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

   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local src = src:type(ctype)
      compareCPUAndCUDATypeTensorArgs(typename, true, src, 'gather', dim, idx)
      compareCPUAndCUDATypeTensorArgs(typename, false, src, 'gather', dim, idx)
   end
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
   local res = torch.FloatTensor(m, n, o):zero()

   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local res, src = res:type(ctype), src:type(ctype)
      compareCPUAndCUDATypeTensorArgs(typename, true, res, 'scatter', dim, idx, src)
      compareCPUAndCUDATypeTensorArgs(typename, false, res, 'scatter', dim, idx, src)
   end
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

   local res = torch.FloatTensor(m, n, o):zero()
   for k, typename in ipairs(typenames) do
      local res = res:type(t2cpu[typename])
      compareCPUAndCUDATypeTensorArgs(typename, true, res, 'scatter', dim, idx, val)
      compareCPUAndCUDATypeTensorArgs(typename, false, res, 'scatter', dim, idx, val)
   end
end

function test.sort()
   for tries = 1, 5 do
      local t = createTestTensor(2 ^ 20)
      local selectdim = chooseInt(1, t:nDimension())
      local dir = chooseInt(1, 2) == 1

      for k, typename in ipairs(typenames) do
          if typename ~= 'torch.CudaByteTensor'
              and typename ~= 'torch.CudaCharTensor'
          and typename ~= 'torch.CudaShortTensor' then
              local ctype = t2cpu[typename]
              local t = t:type(ctype)
              compareCPUAndCUDATypeTensorArgs(typename, nil, t, 'sort', selectdim, dir)
          end
      end
   end

   -- Test a large tensors whose total size exceeds 2^24,
   -- but whose sorting dimension is less than 2^24
   -- Since the sorting mechanism is not guaranteed to be the
   -- same between GPU and CPU, we have to be careful when comparing
   -- the indices
   local t_cpu = torch.FloatTensor(5000, 5000):uniform()
   local t_gpu = t_cpu:cuda()

   local v_cpu, i_cpu = torch.sort(t_cpu, 2)
   local v_gpu, i_gpu = torch.sort(t_gpu, 2)

   -- Values should match exactly, regardless of sorting method
   tester:assert(isEqual(v_cpu, v_gpu), 'value mismatch')

   -- Indices can differ since the sorting method can differ (stable vs. not),
   -- but values should be equivalent after gather
   local gather_cpu = t_cpu:gather(2, i_cpu)
   local gather_gpu = t_gpu:gather(2, i_gpu)

   tester:assert(isEqual(gather_cpu, gather_gpu), 'indices mismatch')

   -- Test a large tensors whose total size exceeds 2^24
   local t_cpu = torch.FloatTensor(2^25):uniform()
   local t_gpu = t_cpu:cuda()

   local v_cpu, i_cpu = torch.sort(t_cpu, 1)
   local v_gpu, i_gpu = torch.sort(t_gpu, 1)

   -- Values should match exactly, regardless of sorting method
   tester:assert(isEqual(v_cpu, v_gpu), 'value mismatch')

   -- Indices can differ since the sorting method can differ (stable vs. not),
   -- but values should be equivalent after gather
   local gather_cpu = t_cpu:gather(1, i_cpu)
   local gather_gpu = t_gpu:gather(1, i_gpu)

   tester:assert(isEqual(gather_cpu, gather_gpu), 'indices mismatch')
end

local function explore(typename, func, t, topk, indices)
   if t:nDimension() == 1 then
      func(typename, t, topk, indices)
   else
      for i = 1, t:size(1) do
         explore(typename, func, t[i], topk[i], indices[i])
      end
   end
end

function test.topk()
   -- need to ensure unique values for index checking, so for the first pass we create Tensors
   -- with sizes less than the maximum range of values for that type
   local counts = {}
   counts['torch.CudaByteTensor'] = 255
   counts['torch.CudaCharTensor'] = 255
   counts['torch.CudaShortTensor'] = 65536
   counts['torch.CudaIntTensor'] = 2 ^ 20
   counts['torch.CudaTensor'] = 2 ^ 20
   counts['torch.CudaLongTensor'] = 2 ^ 20
   counts['torch.CudaDoubleTensor'] =  2 ^ 20
   counts['torch.CudaHalfTensor'] = 32768

   for _, typename in ipairs(typenames) do
      for tries = 1, 5 do
         local t = createTestTensor(counts[typename]):type(typename)
         local dim = chooseInt(1, t:nDimension())
         local dimSize = t:size(dim)
         local dir = chooseInt(1, 2) == 1

         -- Test boundary conditions
         local kTests = {1, dimSize}

         -- and some other random ones
         table.insert(kTests, chooseInt(1, dimSize))
         for i = 1, 2 do
            -- some sizes that fit in our inplace kernel range (the dimSize one
            -- will fall back to Thrust)
            table.insert(kTests, chooseInt(1, math.min(2048, dimSize)))
         end

         for k = 1, #kTests do
            compareCPUAndCUDATypeTensorArgsWithLimit(typename, nil, 1, t, 'topk', kTests[k], dim, dir, true)

            -- verify that indices picked yield topk value in original tensor
            local topk, indices = t:topk(kTests[k], dim, dir, true)
            local verify = function(typename, t, topk, indices)
               t = t:type(t2cpu[typename])
               indices = indices:long()
               topk = topk:type(t2cpu[typename])
               for i = 1, indices:size(1) do
                  tester:assert(t[indices[i]] == topk[i])
               end
            end

            local tt  = t:transpose(dim, t:nDimension())
            local ttk = topk:transpose(dim, topk:nDimension())
            local tti = indices:transpose(dim, indices:nDimension())

            explore(typename, verify, tt, ttk, tti)
         end
      end
   end
end

local function verifyMode1D(tensor)
   -- We cannot rely upon comparing against CPU-Torch as the way it resolves
   -- ties between equal modes and how it picks the corresponding index is not
   -- reliable. Instead we will use apply macros to compute the mode in place in
   -- our code and compare against these results

   -- counts is a table of tensor element -> # of occurrences
   local counts = {}

   -- populate counts by iterating over the elements in the tensor
   tensor:apply(function(x) if counts[x] == nil then counts[x] = 1 else counts[x] = counts[x] + 1 end return x end)

   -- next, calculate the max occurrence in the tensor
   local max = -1;
   for _, count in pairs(counts) do
      if count > max then max = count end
   end

   -- now verify for all the GPU types that 1. the mode picked has max occurrences,
   -- and 2. that the index returned contains that mode

   for _, cudaType in ipairs(typenames) do
      local baseType = t2cpu[cudaType]
      assert(baseType, 'Cannot find baseType for ' .. cudaType)
      local x_cpu = tensor:clone():type(baseType)
      local x_cuda = cloneExactlyToGPUType(x_cpu, nil, t2gpu)

      local modes, indices = x_cuda:mode()

      -- 1D, so should only be a single return
      tester:assert(modes:nElement() == 1, 'mode returned an invalid number of values')
      tester:assert(indices:nElement() == 1, 'mode returned an invalid number of values')
      local mode = modes[1]
      local index = indices[1]

      tester:assert(counts[mode] == max, string.format(
         'Type: %s --> Selected mode of %s which has count of %s, but mode must have %s occurrences',
         cudaType, tostring(mode), tostring(counts[mode]), tostring(max)
      ))
      tester:assert(tensor[index] == mode, string.format(
        'Type: %s --> Selected index of %s which has value %s, but mode is %s',
        cudaType, tostring(index), tostring(tensor[index]), tostring(mode)
      ))
   end
end

local function assertSize(tensor, sizes)
   local valid = true
   if tensor:nDimension() ~= #sizes then
      tester:assert(false, 'tensor dimension mismatch')
   end
   for i, size in ipairs(sizes) do
      if tensor:size(i) ~= size then
         valid = false
      end
   end
   tester:assert(valid, 'tensor size mismatch')
end

local function verifyMode2D(tensor, onlyDim)
   local dims = {}
   if onlyDim ~= nil then
      dims = {onlyDim}
   else
      dims = {1, 2}
   end

   for _, dim in ipairs(dims) do
      -- In the case of a 2D Tensor, we need to calculate the count for each slice
      -- sCounts is a table containing the counts of elements for each slice,
      -- sMax is a table containing the max occurrence for each slice
      local sCounts = {}
      local sMax = {}

      -- First, we use the :split() function to split the Tensor
      -- Suppose we are mode'ing a 5x10 Tensor. If we mode along dim=1,
      -- we have a result Tensor that is 1x10, so we need the counts for
      -- all 10 slices of size=5. So we actually split along dim=2, with
      -- size = 1, to yield 10 5x1 tensors
      local splits = tensor:split(1, dim == 1 and 2 or 1)

      -- next, we iterate over these split Tensors to calculate the mode, as we
      -- did in the 1D case
      for i, slice in pairs(splits) do
         local counts = {}
         slice:apply(function(x) if counts[x] == nil then counts[x] = 1 else counts[x] = counts[x] + 1 end return x end)

         local max = -1;
         for _, count in pairs(counts) do
            if count > max then max = count end
         end

         sCounts[i] = counts;
         sMax[i] = max;
      end

      -- verification pass
      for _, cudaType in ipairs(typenames) do
         local baseType = t2cpu[cudaType]
         assert(baseType, 'Cannot find baseType for ' .. cudaType)
         local x_cpu = tensor:clone():type(baseType)
         local x_cuda = cloneExactlyToGPUType(x_cpu, nil, t2gpu)
         local modes, indices = x_cuda:mode(dim)

         -- 2D, so expect:
         -- (dim = 1) a 1xsize(tensor, dim = 2) tensor
         -- (dim = 2) a size(tensor, dim = 1)x1 tensor

         if dim == 1 then
            assertSize(modes, {1, tensor:size(2)})
            assertSize(indices, {1, tensor:size(2)})
         else
            assertSize(modes, {tensor:size(1), 1})
            assertSize(indices, {tensor:size(1), 1})
         end

         -- we need to run through and verify that all of the modes/indices are valid, for
         -- the results of each slice. First, we squeeze the Tensor, so we can iterate over
         -- both the 1D/2D values in the same manner
         modes = modes:squeeze(dim)
         indices = indices:squeeze(dim)

         -- iterate over each slice, and verify that for each slice the mode selected has
         -- max occurrences, and the index points to the mode
         for i, counts in pairs(sCounts) do
            local max = sMax[i]
            local mode = modes[i]
            local index = indices[i]

            tester:assert(counts[mode] == max, string.format(
               'Type: %s --> Selected mode of %s which has count of %s, but mode must have %s occurrences',
               cudaType, tostring(mode), tostring(counts[mode]), tostring(max)
            ))

            if dim == 1 then
               tester:assert(tensor[index][i] == mode, string.format(
                  'Type: %s --> Selected index of %s which has value %s, but mode is %s',
                  cudaType, tostring(index), tostring(tensor[index][i]), tostring(mode)
               ))
            else
               tester:assert(tensor[i][index] == mode, string.format(
                  'Type: %s --> Selected index of %s which has value %s, but mode is %s',
                  cudaType, tostring(index), tostring(tensor[i][index]), tostring(mode)
               ))
            end
         end
      end
   end
end

local function verifyMode3D(tensor, onlyDim)
    local dims = {}
    if onlyDim ~= nil then
       dims = {onlyDim}
    else
       dims = {1, 2, 3}
    end
    -- In the case of 3D Tensor, we need to calculate the count for each slice,
    -- but this time, we have two layers of depth, for each of the non-mode dims
    -- so sCounts is a multi-level table where sCounts[i][j] is the counts for
    -- (_, i, j), (i, _, j) or (i, j, _) depending on the dim
    local sCounts = {}
    local sMax = {}

    -- Suppose we have a 2x3x4 Tensor T:
    -- (1, .., ..),    (2, .., ..)
    -- [1, 2, 3, 4]    [3, 2, 2, 4]
    -- [5, 6, 7, 8]    [5, 6, 8, 7]
    -- [9, 10, 11, 12] [1, 10, 11, 1]
    --
    -- Then for dim = 1, we need counts to be a multi-level table (3x4xcounts)
    --                2                                           (2x4xcounts)
    --                3                                           (2x3xcounts)
    --
    -- Results: dim = 1
    -- {1:
    --    {1:
    --       1 --> 1,
    --       3 --> 1,
    --     2:
    --       2 --> 2,
    --     3:
    --       2 --> 1,
    --       3 --> 1,
    --     4:
    --       4 --> 2,
    --    },
    -- {2:
    --    {1:
    --       5 --> 2,
    --       ...

    -- used to set loop bounds and indexing to construct the above table using the loop below
    local dbounds = {
      {tensor:size(2), tensor:size(3), tensor:size(1)},
      {tensor:size(1), tensor:size(3), tensor:size(2)},
      {tensor:size(1), tensor:size(2), tensor:size(3)}}
    local dfuncs = {
      function(tensor, i, j, k) return tensor[k][i][j] end,
      function(tensor, i, j, k) return tensor[i][k][j] end,
      function(tensor, i, j, k) return tensor[i][j][k] end,
    }

    -- loop...
    for d, bounds in ipairs(dbounds) do
      sCounts[d] = {}
      sMax[d] = {}
      for i = 1, bounds[1] do
        sCounts[d][i] = {}
        sMax[d][i] = {}
        for j = 1, bounds[2] do
           sCounts[d][i][j] = {}
           sMax[d][i][j] = {}
           for k = 1, bounds[3] do
             local v = dfuncs[d](tensor, i, j, k)
             if sCounts[d][i][j][v] == nil then
                sCounts[d][i][j][v] = 1
             else
                sCounts[d][i][j][v] = sCounts[d][i][j][v] + 1
             end

             local max = -1
             for _, count in pairs(sCounts[d][i][j]) do
                if count > max then max = count end
             end
             sMax[d][i][j] = max
           end -- k
        end -- k
      end -- j
    end -- d


   -- verification pass
   for _, dim in ipairs(dims) do
      for _, cudaType in ipairs(typenames) do
         local baseType = t2cpu[cudaType]
         assert(baseType, 'Cannot find baseType for ' .. cudaType)
         local x_cpu = tensor:clone():type(baseType)
         local x_cuda = cloneExactlyToGPUType(x_cpu, nil, t2gpu)
         local modes, indices = x_cuda:mode(dim)

         if dim == 1 then
            assertSize(modes, {1, tensor:size(2), tensor:size(3)})
            assertSize(indices, {1, tensor:size(2), tensor:size(3)})
         elseif dim == 2 then
            assertSize(modes, {tensor:size(1), 1, tensor:size(3)})
            assertSize(indices, {tensor:size(1), 1, tensor:size(3)})
         else
            assertSize(modes, {tensor:size(1), tensor:size(2), 1})
            assertSize(indices, {tensor:size(1), tensor:size(2), 1})
         end

         -- squeeze on mode dim
         modes = modes:squeeze(dim)
         indices = indices:squeeze(dim)

         -- iterate over slices
         for i, js in pairs(sCounts[dim]) do
            for j, counts in pairs(js) do
               local max = sMax[dim][i][j]
               local mode = modes[i][j]
               local index = indices[i][j]

               tester:assert(counts[mode] == max, string.format(
                  'Type: %s --> Selected mode of %s which has count of %s, but mode must have %s occurrences',
                  cudaType, tostring(mode), tostring(counts[mode]), tostring(max)
               ))

               if dim == 1 then
                  tester:assert(tensor[index][i][j] == mode, string.format(
                     'Type: %s --> Selected index of %s which has value %s, but mode is %s',
                     cudaType, tostring(index), tostring(tensor[index][i][j]), tostring(mode)
                  ))
               elseif dim == 2 then
                  tester:assert(tensor[i][index][j] == mode, string.format(
                     'Type: %s --> Selected index of %s which has value %s, but mode is %s',
                     cudaType, tostring(index), tostring(tensor[i][index][j]), tostring(mode)
                  ))
               else
                  tester:assert(tensor[i][j][index] == mode, string.format(
                     'Type: %s --> Selected index of %s which has value %s, but mode is %s',
                     cudaType, tostring(index), tostring(tensor[i][j][index]), tostring(mode)
                  ))
               end

            end -- j
         end --i
      end -- tensor type
   end -- dim
end

function test.mode()
    -- Tests for 1D Tensors

    -- Single-element Tensor
    local input = torch.FloatTensor({1})
    verifyMode1D(input)

    -- Tensor of all the same values
    local input = torch.FloatTensor(10):fill(1)
    verifyMode1D(input)

    -- Tensor with a unique range of values
    local input = torch.FloatTensor({4, 3, 6, 8, 2, 1})
    verifyMode1D(input)

    -- Handles ties when there are two things with equal counts
    local input = torch.FloatTensor({2, 2, 1, 1})
    verifyMode1D(input)

    -- Big Range of Values: (4 is the mode)
    local input = torch.FloatTensor({
        1, 4, 4, 4, 4, 1, 1, 2, 2, 2, 3, 4, 5, 5, 4, 4, 4, 4, 4, 4,
        2, 2, 1, 1, 2, 3, 4, 4, 4, 4, 2, 3, 4, 4, 3, 2, 1, 2, 3, 4})
    verifyMode1D(input)

    -- Larger Example
    local input = torch.FloatTensor(1000):apply(function(x) return torch.random(1, 10) end)
    verifyMode1D(input)

    -- verify input is unchanged
    local input = torch.FloatTensor({4, 3, 6, 8, 2, 1})
    local same = torch.FloatTensor({4, 3, 6, 8, 2, 1})
    torch.mode(input)
    tester:assertTensorEq(input, same, 0, 'cutorch mode modified input')

    -- Tests for 2D Tensors

    -- Tensor of all the same values
    local input = torch.FloatTensor(3, 4):fill(1)
    verifyMode2D(input)

    -- Tensor with a unique range of values
    local input = torch.FloatTensor({{2,  3,  5, 7},
                                     {1, 10, 17, 6},
                                     {0, 22, 14, 9}})
    verifyMode2D(input)

    -- Consistency between ties when there are two things with equal counts
    local input = torch.FloatTensor({{2,  2,  3, 3},
                                     {1,  1,  3, 3},
                                     {2,  2,  1, 1},
                                     {1,  1,  1, 1}})
    verifyMode2D(input)

    -- Larger example
    local input = torch.FloatTensor(50, 100):apply(function(x) return torch.random(1, 10) end)
    verifyMode2D(input)

    -- Tests for 3D Tensors

    -- Tensor of all the same values
    local input = torch.FloatTensor(2, 4, 5):fill(1)
    verifyMode3D(input)

    -- Tensor with a unique range of values
    local input = torch.FloatTensor(
        {
            {{2,  3,  5, 7},
             {1, 10, 17, 6},
             {0, 22, 14, 9}},

            {{32, 88, 25,   4},
             {21, 78, 57, 111},
             {15, 68, 64, 117}}
        }
    )
    verifyMode3D(input)

    -- Handles ties when there are two things with equal counts
    local input = torch.FloatTensor(
        {
            {{2,  2,  3, 3},
             {1,  1,  3, 3},
             {2,  2,  1, 1},
             {1,  1,  1, 1}},

            {{3,  3,  4, 4},
             {2,  2,  4, 4},
             {3,  3,  2, 2},
             {2,  2,  2, 2}},
        }
    )
    verifyMode3D(input)

    -- Larger example
    local input = torch.FloatTensor(14, 22, 32):apply(function(x) return torch.random(1, 10) end)
    verifyMode3D(input)
end

function test.bigmode()
    -- Examples that overflow fused-kernel
    local input = torch.IntTensor(16384):apply(function(x) return torch.random(1, 100) end)
    verifyMode1D(input)

    local input = torch.FloatTensor(4096, 4):fill(1)
    verifyMode2D(input, 1)

    local input = torch.FloatTensor(4, 4096):fill(1)
    verifyMode2D(input, 2)

    local input = torch.FloatTensor(2, 2, 4096):fill(1)
    verifyMode3D(input, 3)

    local input = torch.FloatTensor(2, 4096, 2):fill(1)
    verifyMode3D(input, 2)

    local input = torch.FloatTensor(4096, 2, 2):fill(1)
    verifyMode3D(input, 1)
end

function test.cat()
   for k, typename in ipairs(typenames) do
      for dim = 1, 3 do
         local x = torch.Tensor(13, minsize, minsize):uniform()
            :type(typename):transpose(1, dim)
         local y = torch.Tensor(17, minsize, minsize):uniform()
            :type(typename):transpose(1, dim)
         local mx = torch.cat(x, y, dim)
         tester:assertTensorEq(mx:narrow(dim, 1, 13), x, 0, 'torch.cat value')
         tester:assertTensorEq(mx:narrow(dim, 14, 17), y, 0, 'torch.cat value')

         local mxx = torch.Tensor():type(typename)
         torch.cat(mxx, x, y, dim)
         tester:assertTensorEq(mx, mxx, 0, 'torch.cat value')

         local x = torch.CudaTensor(1, 2, 3):uniform()
         local y = torch.CudaTensor()
         local mx = torch.cat(x,y,dim)
         tester:asserteq(mx:size(1),1,'torch.cat size')
         tester:asserteq(mx:size(2),2,'torch.cat size')
         tester:asserteq(mx:size(3),3,'torch.cat size')
         tester:assertTensorEq(mx, x, 0, 'torch.cat value')

         local x = torch.CudaTensor()
         local y = torch.CudaTensor()
         local mx = torch.cat(x,y,dim)
         tester:asserteq(mx:dim(),0,'torch.cat dim')
      end
   end
end

function test.catNoDim()
   for k, typename in ipairs(typenames) do
      local a
      local b
      local c

      a = torch.Tensor(minsize):uniform():type(typename)
      b = torch.Tensor(minsize):uniform():type(typename)
      c = torch.cat(a, b)
      tester:assertTensorEq(c:narrow(1, 1, minsize), a, 0, 'torch.cat value')
      tester:assertTensorEq(c:narrow(1, minsize + 1, minsize), b, 0, 'torch.cat value')

      a = torch.Tensor(1, minsize):uniform():type(typename)
      b = torch.Tensor(1, minsize):uniform():type(typename)
      c = torch.cat(a, b)
      tester:assertTensorEq(c:narrow(2, 1, minsize), a, 0, 'torch.cat value')
      tester:assertTensorEq(c:narrow(2, minsize + 1, minsize), b, 0, 'torch.cat value')

      a = torch.Tensor(10, minsize):uniform():type(typename)
      b = torch.Tensor(10, minsize):uniform():type(typename)
      c = torch.cat(a, b)
      tester:assertTensorEq(c:narrow(2, 1, minsize), a, 0, 'torch.cat value')
      tester:assertTensorEq(c:narrow(2, minsize + 1, minsize), b, 0, 'torch.cat value')
   end
end

function test.catArray()
   for k, typename in ipairs(typenames) do
      for dim = 1, 3 do
         local x = torch.Tensor(13, minsize, minsize):uniform()
            :type(typename):transpose(1, dim)
         local y = torch.Tensor(17, minsize, minsize):uniform()
            :type(typename):transpose(1, dim)
         local z = torch.Tensor(19, minsize, minsize):uniform()
            :type(typename):transpose(1, dim)

         local mx = torch.cat({x, y, z}, dim)
         tester:assertTensorEq(mx:narrow(dim, 1, 13), x, 0, 'torch.cat value')
         tester:assertTensorEq(mx:narrow(dim, 14, 17), y, 0, 'torch.cat value')
         tester:assertTensorEq(mx:narrow(dim, 31, 19), z, 0, 'torch.cat value')

         local mxx = torch.Tensor():type(typename)
         torch.cat(mxx, {x, y, z}, dim)
         tester:assertTensorEq(mx, mxx, 0, 'torch.cat value')

         local x = torch.CudaTensor(1, 2, 3):uniform()
         local y = torch.CudaTensor()
         local mx = torch.cat({x,y},dim)
         tester:asserteq(mx:size(1),1,'torch.cat size')
         tester:asserteq(mx:size(2),2,'torch.cat size')
         tester:asserteq(mx:size(3),3,'torch.cat size')
         tester:assertTensorEq(mx, x, 0, 'torch.cat value')

         local x = torch.CudaTensor()
         local y = torch.CudaTensor()
         local mx = torch.cat({x,y},dim)
         tester:asserteq(mx:dim(),0,'torch.cat dim')
      end
   end
end

-- designed to specifically hit the batched kernel for catArray
function test.catArrayBatched()
    local batchSizes = {2, 16, 128, 1024, 4096}
    for _, batchSize in ipairs(batchSizes) do
        -- first, batches for 1D Tensors
        local tensors = {}
        for i = 1, batchSize do
            table.insert(tensors, torch.CudaTensor(1024):uniform())
        end
        local mx = torch.cat(tensors, 1)
        local offset = 1
        for i = 1, batchSize do
            tester:assertTensorEq(mx:narrow(1, offset, tensors[i]:size(1)), tensors[i], 0, 'torch.carArrayBatched value')
            offset = offset + tensors[i]:size(1)
        end

        -- next, 2D Tensors
        tensors = {}
        for i = 1, batchSize do
            table.insert(tensors, torch.CudaTensor(1, 1024):uniform())
        end
        -- across dim = 1 (row-wise concatentation)
        mx = torch.cat(tensors, 1)
        offset = 1
        for i = 1, batchSize do
            tester:assertTensorEq(mx:narrow(1, offset, tensors[i]:size(1)), tensors[i], 0, 'torch.carArrayBatched value')
            offset = offset + tensors[i]:size(1)
        end
        tensors = {}
        for i = 1, batchSize do
            table.insert(tensors, torch.CudaTensor(128, 128):uniform())
        end
        -- across dim = 2 (column-wise concatentation)
        mx = torch.cat(tensors, 2)
        offset = 1
        for i = 1, batchSize do
            tester:assertTensorEq(mx:narrow(2, offset, tensors[i]:size(2)), tensors[i], 0, 'torch.carArrayBatched value')
            offset = offset + tensors[i]:size(2)
        end
    end

    -- one giant copy
    local a = torch.CudaTensor(4096, 4096):uniform()
    local b = torch.CudaTensor(4096, 4096):uniform()
    local mx = torch.cat({a, b}, 1)
    tester:assertTensorEq(mx:narrow(1, 1, 4096), a, 0, 'torch.carArrayBatched value')
    tester:assertTensorEq(mx:narrow(1, 4097, 4096), b, 0, 'torch.carArrayBatched value')

    -- output Tensor is non-contiguous
    local notcontig = torch.CudaTensor(5, 4):t():uniform()
    local a = torch.CudaTensor(2, 5):uniform()
    local b = torch.CudaTensor(1, 5):uniform()
    local c = torch.CudaTensor(1, 5):uniform()

    torch.cat(notcontig, {a, b, c}, 1)
    tester:assertTensorEq(notcontig:narrow(1, 1, 2), a, 0, 'torch.carArrayBatched value')
    tester:assertTensorEq(notcontig:narrow(1, 3, 1), b, 0, 'torch.carArrayBatched value')
    tester:assertTensorEq(notcontig:narrow(1, 4, 1), c, 0, 'torch.carArrayBatched value')
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
   for i = 1, iter do
      for stream = numStreams, 1, -1 do
         cutorch.setStream(stream)
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
   local iter = 80 + torch.random(10)
   local numStreams = torch.random(10)
   cutorch.reserveStreams(numStreams + 1)

   -- Create scratch space on the last device to receive all results
   -- `tmpResults` and `results` will be operated on in `numStreams + 1`
   cutorch.setDevice(numDevices)
   cutorch.setStream(numStreams + 1)
   local tmpResults = {}
   local results = torch.CudaTensor(size):zero()

   for dev = 1, numDevices - 1 do
      local tmpResultsPerDevice = {}
      for stream = 1, numStreams do
         table.insert(tmpResultsPerDevice, torch.CudaTensor(size):zero())
      end

      table.insert(tmpResults, tmpResultsPerDevice)
   end

   -- In order to test isolating the one-way barrier below, sync all the work
   -- above so we know the `zero()` is complete.
   cutorch.streamSynchronize(numStreams + 1)

   -- Allocate data on all devices (except the last)
   local tensors = {}

   for dev = 1, numDevices - 1 do
      cutorch.setDevice(dev)
      local tensorsPerDevice = {}

      for stream = 1, numStreams do
         cutorch.setStream(stream)
         table.insert(tensorsPerDevice, torch.CudaTensor(size):zero())
      end

      table.insert(tensors, tensorsPerDevice)
   end

   -- Queue work to all streams, all devices (except the last)
   for i = 1, iter do
      for dev = 1, numDevices - 1 do
         cutorch.setDevice(dev)
         for stream = 1, numStreams do
            cutorch.setStream(stream)
            tensors[dev][stream]:add(1)
         end
      end
   end

   -- Copy back to device `numDevices`
   for dev = 1, numDevices - 1 do
      cutorch.setDevice(dev)
      for stream = 1, numStreams do
         cutorch.setStream(stream)

         -- These copies will be ordered in the source stream (dev, stream), but
         -- tmpResults is on device `numDevices`.
         tmpResults[dev][stream]:copy(tensors[dev][stream])

         -- We will wait on the above copy to complete in the dest too
         cutorch.streamWaitForMultiDevice(numDevices, numStreams + 1, {[dev]={stream}})

         -- Note that because the copy is ordered in (dev, stream), we are free
         -- to modify the value after issuing the above copy.
         tensors[dev][stream]:zero()
      end
   end

   -- Sum up the results
   cutorch.setDevice(numDevices)
   cutorch.setStream(numStreams + 1)

   for dev = 1, numDevices - 1 do
      for stream = 1, numStreams do
         results:add(tmpResults[dev][stream])
      end
   end

   tester:asserteq(results:min(), iter * numStreams * (numDevices - 1))

   -- return to default device/stream
   cutorch.setDevice(1)
   cutorch.setStream(0)
   results = nil
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
   for stream = numStreams, 1, -1 do
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
   for i = 1, iter do
      for dev = 1, numDevices do
         cutorch.setDevice(dev)
         for stream = 1, numStreams do
            cutorch.setStream(stream)
            tensors[dev][stream]:add(1)
         end
      end
   end

   -- Create an all-way barrier
   cutorch.streamBarrierMultiDevice(waitingFor)

   -- All-to-all copy (done in stream 1 on each device)
   for dev = 1, numDevices do
      cutorch.setDevice(dev)
      cutorch.setStream(1)

      for otherDev = 1, numDevices do
         for otherStream = 1, numStreams do
            -- This copy is ordered in the source (otherDev, stream 1)
            -- which produced the value.
            -- (dev, stream 1) on all devices is complete due to the all-way
            -- barrier above.
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
            -- Since the copy above is ordered in stream (otherDev, 1),
            -- we need to wait for its completion
            if dev ~= otherDev then
               cutorch.streamWaitForMultiDevice(dev, 1, {[otherDev]={1}})
            end

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

function test.cudaEvent()
   cutorch.reserveStreams(2)
   cutorch.setStream(1)

   local t1 = torch.CudaTensor(100000000):zero()
   local t2 = torch.CudaTensor(1):zero()

   local t1View = t1:narrow(1, 100000000, 1)
   t1:fill(1)
   -- Event is created here
   local event = cutorch.Event()

   cutorch.setStream(2)

   -- assert below will fail without this
   event:waitOn()
   t2:copy(t1View)
   tester:asserteq(t2[1], 1)

   -- revert to default stream
   cutorch.setStream(0)
end

function test.cudaHostTensor()
  local t = cutorch.createCudaHostTensor(3, 4, 5)
  tester:assertTableEq(t:size():totable(), {3, 4, 5})

  local u = torch.Tensor(4, 5, 6)
  local v = cutorch.createCudaHostTensor(u:size())
  tester:assertTableEq(u:size():totable(), v:size():totable())

  local w = cutorch.createCudaHostTensor()
  tester:assert(w:storage() ~= nil, 'Empty CUDA host tensor must have a storage')
  tester:asserteq(w:nElement(), 0, 'Expected an empty tensor')
end

function test.kernelP2PAccess()
   -- We can only test direct kernel p2p access if we have multiple devices
   -- and p2p enabled
   if cutorch.getDeviceCount() < 2 then
      return
   end

   if cutorch.getPeerToPeerAccess(1, 2) then
      -- We should be on device 1 anyways, but just make sure
      cutorch.setDevice(1)
      local a = torch.CudaTensor(8):zero()
      local b = nil
      cutorch.withDevice(2, function() b = torch.CudaTensor(8):fill(1) end)

      local expected = false

      -- a is on device 1, b is on device 2, so this is a kernel p2p access
      local function tryAdd()
         local ok, err = pcall(function() a:add(b) end)
         tester:assert(ok == expected)
      end

      -- By default, direct kernel p2p access should be an error
      cutorch.setKernelPeerToPeerAccess(false)
      cutorch.withDevice(1, tryAdd)
      tester:asserteq(a:sum(), 0)

      -- Now enable and try again
      cutorch.setKernelPeerToPeerAccess(true)
      expected = true
      cutorch.withDevice(1, tryAdd)
      tester:asserteq(a:sum(), 8)

      a:zero()

      -- Turn it back off and check again
      cutorch.setKernelPeerToPeerAccess(false)
      expected = false
      cutorch.withDevice(1, tryAdd)
      tester:asserteq(a:sum(), 0)
   end
end

if os.getenv('THC_CACHING_ALLOCATOR') ~= '0' then
   local function getCyclesPerMs()
      cutorch.synchronize()
      local t = torch.Timer()
      cutorch._sleep(1e6)
      cutorch.synchronize()
      return 1e6 / (t:time().real * 1000)
   end

   function test.cachedPinnedMemory()
      local cyclesPerMs = getCyclesPerMs()

      -- check that allocations are re-used after deletion
      local t = cutorch.createCudaHostTensor({1})
      local ptr = t:data()
      t = nil; collectgarbage()
      t = cutorch.createCudaHostTensor({1})
      tester:asserteq(t:data(), ptr, 'allocation not reused')

      -- check that the allocation is not re-used if it's in-use by a copy
      gpuTensor = torch.CudaTensor({0})
      cutorch._sleep(50 * cyclesPerMs)  -- delay the copy
      gpuTensor:copyAsync(t)
      t = nil; collectgarbage()
      t = cutorch.createCudaHostTensor({1})
      tester:assertne(t:data(), ptr, 'allocation re-used too soon')
   end

   function test.cachedPinnedMemoryMultiGPU()
      local device_count = cutorch.getDeviceCount()
      if device_count < 2 then
         return
      end

      local cyclesPerMs = getCyclesPerMs()
      local t = cutorch.createCudaHostTensor(1)
      local ptr = t:data()
      t[1] = 1

      local gpu_tensor1 = torch.CudaTensor({0})

      cutorch.setDevice(2)
      local gpu_tensor2 = torch.CudaTensor({0})
      cutorch._sleep(50 * cyclesPerMs)  -- delay the copy
      gpu_tensor2:copyAsync(t)

      cutorch.setDevice(1)
      t = nil; collectgarbage();
      t = cutorch.createCudaHostTensor(1)
      tester:assertne(t:data(), ptr, 'allocation re-used too soon')
   end

end

-- unfortunately, torch.Tester() forgot setUp and tearDown functions.
-- It would be nice to fix torch.Tester() eventually.
local function setUp()
  cutorch.setDevice(1)
  checkHalf()
end

local test_ = torch.TestSuite()
for k,v in pairs(test) do
  test_[k] = function()
    setUp()
    v()
  end
end
test = test_

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
end

if runtests then
   cutorch.test()
   os.exit(#tester.errors == 0 and 0 or 1)
end
return test
