cutorch
=======

Cutorch provides a CUDA backend for torch7.

Cutorch provides the following:

- a new tensor type: `torch.CudaTensor` that acts like `torch.FloatTensor`, but all it's operations are on the GPU. Most of the tensor operations are supported by cutorch. There are a few missing ones, which are being implemented. The missing list can be found here: https://github.com/torch/cutorch/issues/70
- several other GPU tensor types, with limited functionality. Currently limited to copying/conversion, and several indexing and shaping operations.
- `cutorch.*` - Functions to set/get GPU, get device properties, memory usage, set/get low-level streams, set/get random number generator's seed, synchronization etc. They are described in more detail below.

### torch.CudaTensor
This new tensor type behaves exactly like a `torch.FloatTensor`, but has a couple of extra functions of note:
- `t:getDevice()` - Given a CudaTensor `t`, you can call :getDevice on it to find out the GPU ID on which the tensor memory is allocated.

### Other CUDA tensor types
Most other (besides float) CPU torch tensor types now have a cutorch equivalent, with similar names:

- `torch.CudaDoubleTensor`
- `torch.CudaByteTensor`
- `torch.CudaCharTensor`
- `torch.CudaIntTensor`
- `torch.CudaShortTensor`
- `torch.CudaLongTensor`
- and `torch.CudaHalfTensor` when supported as indicated by `cutorch.hasHalf`; these are half-precision (16-bit) floats.

**Note:** these are currently limited to copying/conversion, and several indexing and shaping operations (e.g. `narrow`, `select`, `unfold`, `transpose`).

###`cutorch.*` API
- `cutorch.synchronize()` : All of the CUDA API is asynchronous (barring a few functions), which means that you can queue up operations. To wait for the operations to finish, you can issue `cutorch.synchronize()` in your code, when the code waits for all GPU operations on the current GPU to finish. WARNING: synchronizes the CPU host with respect to the current device (as per `cutorch.getDevice()`) only.
- `cutorch.synchronizeAll()` : Same as `cutorch.synchronize()` except synchronizes the CPU host with all visible GPU devices in the system. Equivalent to calling `cutorch.synchronize()` once per each device.
- `cutorch.setDevice(i)` : If one has multiple-GPUs, you can switch the default GPU (to allocate CUDA tensors and do operations). The GPU IDs are 1-indexed, so having 4 GPUs means, you can setDevice(1), setDevice(2), setDevice(3), setDevice(4).
- `idx = cutorch.getDevice()` : Returns the currently set GPU device index.
- `count = cutorch.getDeviceCount()` : Gets the number of available GPUs.
- `freeMemory, totalMemory = cutorch.getMemoryUsage(devID)` : Gets the total and free memory in bytes for the given device ID.
- `cutorch.seed([devID])` - Sets and returns a random seed for the current or specified device.
- `cutorch.seedAll()` - Sets and returns a random seed for all available GPU devices.
- `cutorch.initialSeed([devID])` - Returns the seed for the current or specified device
- `cutorch.manualSeed(seed [, device])` - Sets a manually specified RNG seed for the current or specified device
- `cutorch.manualSeedAll(seed)` - Sets a manually specified RNG seed for all available GPUs
- `cutorch.getRNGState([device])` - returns the current RNG state in the form of a byte tensor, for the current or specified device.
- `cutorch.setRNGState(state [, device])` - Sets the RNG state from a previously saved state, on the current or specified device.
- `cutorch.getState()` - Returns the global state of the cutorch package. This state is not for users, it stores the raw RNG states, cublas handles and other thread and device-specific stuff.
- `cutorch.withDevice(devID, f)` - This is a convenience for multi-GPU code, that takes in a device ID as well as a function f. It switches cutorch to the new device, executes the function f, and switches back cutorch to the original device.
- `cutorch.createCudaHostTensor([...])` - Allocates a `torch.FloatTensor` of [host-pinned memory](https://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/), where dimensions can be given as an argument list of sizes or a `torch.LongStorage`.

#### Low-level streams functions (dont use this as a user, easy to shoot yourself in the foot):
- `cutorch.reserveStreams(n [, nonblocking])`: creates n user streams for use on every device. NOTE: stream index `s` on device 1 is a different cudaStream_t than stream `s` on device 2. Takes an optional non-blocking flag; by default, this is assumed to be false. If true, then the stream is created with cudaStreamNonBlocking.
- `n = cutorch.getNumStreams()`: returns the number of user streams available on every device. By `default`, this is `0`, meaning only the default stream (stream 0) is available.
- `cutorch.setStream(n)`: specifies that the current stream active for the current device (or any other device) is `n`. This is preserved across device switches. 1-N are user streams, `0` is the default stream.
- `n = cutorch.getStream()`: returns the current stream active. By default, returns `0`.
- `cutorch.setDefaultStream()`: an alias for `cutorch.setStream(0)`
- `cutorch.streamWaitFor(streamWaiting, {streamsToWaitOn...})`: A 1-to-N-way barrier. `streamWaiting` will wait for the list of streams specified to finish executing all kernels/events/barriers. Does not block any of the streamsToWaitOn. Current device only.
- `cutorch.streamWaitForMultiDevice(deviceWaiting, streamWaiting, {[device]={streamsToWaitOn...}...})`: (deviceWaiting, streamWaiting) will wait on the list of (`device`, `streams`...) pairs; handles single or multiple device. `cutorch.streamWaitForMultiDevice, a, b, {[a]={streams...}})` is equivalent to `cutorch.setDevice(a); cutorch.streamWaitFor(b, {streams...})`.
- `cutorch.streamBarrier({streams...})`: an N-to-N-way barrier between all the streams; all streams will wait for the completion of all other streams on the current device only. More efficient than creating the same N-to-N-way dependency via `streamWaitFor`.
- `cutorch.streamBarrierMultiDevice({[device]={streamsToWaitOn...}...})`: As with streamBarrier but allows barriers between streams on arbitrary devices. Creates a cross-device N-to-N-way barrier between all (device, stream) values listed.
- `cutorch.streamSynchronize(stream)`: equivalent to `cudaStreamSynchronize(stream)` for the current device. Blocks the CPU until stream completes its queued kernels/events.
- `cutorch.setPeerToPeerAccess(dev, devToAccess, f)`: explicitly enable (`f` true) or disable p2p access (`f` false) from `dev` accessing memory on `devToAccess`. Affects copy efficiency (if disabled, copies will be d2d rather than p2p; i.e., the CPU intermediates), and affects kernel p2p access as well. Can only be enabled if the underlying hardware supports p2p access. p2p access is enabled by default for all pairs of devices if the underlying hardware supports it.
- `cutorch.getPeerToPeerAccess(dev, devToAccess)`: returns whether or not p2p access is currently enabled or disabled, for reasons of a prior call of `setPeerToPeerAccess` or underlying hardware support.
- `cutorch.setKernelPeerToPeerAccess(f)`: by default, kernels running on one device cannot directly access memory on another device. This is a check imposed by cutorch, to prevent synchronization and performance issues. To disable the check, call this with `f` true. Kernel p2p access is actually only allowed for a pair of devices if both this is true and the underlying `getPeerToPeerAccess` for the pair involved is true.
- `cutorch.getKernelPeerToPeerAccess()`: returns whether or not kernel p2p checks are enabled or disabled.

##### Common Examples
Transfering a FloatTensor `src` to the GPU:
```lua
dest = src:cuda() -- dest is on the current GPU
```

Allocating a tensor on a given GPU:
Allocate `src` on GPU 3
```lua
cutorch.setDevice(3)
src = torch.CudaTensor(100)
```

Copying a CUDA tensor from one GPU to another:
Given a tensor called `src` on GPU 1, if you want to create it's clone on GPU 2, then:

```lua
cutorch.setDevice(2)
local dest = src:clone()
```

OR

```lua
local dest
cutorch.withDevice(2, function() dest = src:clone() end)
```
