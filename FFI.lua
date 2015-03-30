if jit then

   local ffi = require 'ffi'

   local cdefs = [[
typedef struct THCState
{
  struct THCRNGState* rngState;
  struct THCBlasState* blasState;
  struct cudaDeviceProp* deviceProperties;
} THCState;

typedef struct THCudaStorage
{
    float *data;
    long size;
    int refcount;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
} THCudaStorage;

typedef struct THCudaTensor
{
    long *size;
    long *stride;
    int nDimension;

    THCudaStorage *storage;
    long storageOffset;
    int refcount;

    char flag;

} THCudaTensor;
]]
   ffi.cdef(cdefs)

   local Storage = torch.getmetatable('torch.CudaStorage')
   local Storage_tt = ffi.typeof('THCudaStorage**')

   rawset(Storage, "cdata", function(self) return Storage_tt(self)[0] end)
   rawset(Storage, "data", function(self) return Storage_tt(self)[0].data end)
   -- Tensor
   local Tensor = torch.getmetatable('torch.CudaTensor')
   local Tensor_tt = ffi.typeof('THCudaTensor**')

   rawset(Tensor, "cdata", function(self) return Tensor_tt(self)[0] end)

   rawset(Tensor, "data",
          function(self)
             self = Tensor_tt(self)[0]
             return self.storage ~= nil and self.storage.data + self.storageOffset or nil
          end
   )

end
