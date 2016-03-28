local ok, ffi = pcall(require, 'ffi')
if ok then
   local unpack = unpack or table.unpack
   local cdefs = [[
typedef struct CUstream_st *cudaStream_t;

struct cublasContext;
typedef struct cublasContext *cublasHandle_t;
typedef struct CUhandle_st *cublasHandle_t;

typedef struct _THCCudaResourcesPerDevice {
  cudaStream_t* streams;
  cublasHandle_t* blasHandles;
  size_t scratchSpacePerStream;
  void** devScratchSpacePerStream;
} THCCudaResourcesPerDevice;


typedef struct THCState
{
  struct THCRNGState* rngState;
  struct cudaDeviceProp* deviceProperties;
  cudaStream_t currentStream;
  cublasHandle_t currentBlasHandle;
  THCCudaResourcesPerDevice* resourcesPerDevice;
  int numDevices;
  int numUserStreams;
  int numUserBlasHandles;
  int currentPerDeviceStream;
  int currentPerDeviceBlasHandle;
  struct THAllocator* cudaHostAllocator;
} THCState;

cudaStream_t THCState_getCurrentStream(THCState *state);

]]

   local CudaTypes = {
      {'float', ''},
      {'unsigned char', 'Byte'},
      {'char', 'Char'},
      {'short', 'Short'},
      {'int', 'Int'},
      {'long','Long'},
      {'double','Double'},
  }
  if cutorch.hasHalf then
      table.insert(CudaTypes, {'half','Half'})
  end

   for _, typedata in ipairs(CudaTypes) do
      local real, Real = unpack(typedata)
      local ctype_def = [[
typedef struct THCStorage
{
    real *data;
    long size;
    int refcount;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
    struct THCStorage *view;
} THCStorage;

typedef struct THCTensor
{
    long *size;
    long *stride;
    int nDimension;

    THCStorage *storage;
    long storageOffset;
    int refcount;

    char flag;

} THCTensor;
]]

      ctype_def = ctype_def:gsub('real',real):gsub('THCStorage','THCuda'..Real..'Storage'):gsub('THCTensor','THCuda'..Real..'Tensor')
      cdefs = cdefs .. ctype_def
   end
   if cutorch.hasHalf then
      ffi.cdef([[
typedef struct {
    unsigned short x;
} __half;
typedef __half half;
      ]])
   end
   ffi.cdef(cdefs)

   for _, typedata in ipairs(CudaTypes) do
      local real, Real = unpack(typedata)
      local Storage = torch.getmetatable('torch.Cuda' .. Real .. 'Storage')
      local Storage_tt = ffi.typeof('THCuda' .. Real .. 'Storage**')

      rawset(Storage, "cdata", function(self) return Storage_tt(self)[0] end)
      rawset(Storage, "data", function(self) return Storage_tt(self)[0].data end)
      -- Tensor
      local Tensor = torch.getmetatable('torch.Cuda' .. Real .. 'Tensor')
      local Tensor_tt = ffi.typeof('THCuda' .. Real .. 'Tensor**')

      rawset(Tensor, "cdata", function(self) return Tensor_tt(self)[0] end)

      rawset(Tensor, "data",
             function(self)
                self = Tensor_tt(self)[0]
                return self.storage ~= nil and self.storage.data + self.storageOffset or nil
             end
      )
   end

end
