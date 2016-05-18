-- it has always been supported to load cutorch first, and then cltorch
-- but loading cltorch first, and then cutorch, causes cutorch to clobber cltorch
-- this test checks that it does not

-- note that cutorch must NOT have been `require`d prior to running this test, otherwise this
-- test is no longer valid

assert(cutorch == nil)
assert(cltorch == nil)

require 'cltorch'
assert(cltorch ~= nil)
require 'cutorch'
assert(cutorch ~= nil)

cla = torch.ClTensor(3,2):uniform()
print('cla', cla)

cua = torch.CudaTensor(3,2):uniform()
print('cua', cua)

claf = cla:float()
cuaf = cua:float()

cla = claf:cl()
cua = cuaf:cuda()

-- if we got to here, we're probably good :-)

