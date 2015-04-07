cutorch
=======

A CUDA backend for Torch7

Low-level streams functions (dont use this as a user, easy to shoot yourself in the foot):
- `cutorch.reserveStreams(n)`: creates n user streams for use on every device.
- `n = cutorch.getNumStreams()`: returns the number of user streams available on every device. By `default`, this is `0`, meaning only the default stream (stream 0) is available.
- `cutorch.setStream(n)`: specifies that the current stream active for the current device (or any other device) is `n`. This is preserved across device switches. 1-N are user streams, `0` is the default stream.
- `n = cutorch.getStream()`: returns the current stream active. By default, returns `0`.
- `cutorch.setDefaultStream()`: an alias for `cutorch.setStream(0)`
- `cutorch.streamWaitFor(streamWaiting, {streamsToWaitOn...})`: A 1-to-N-way barrier. `streamWaiting` will wait for the list of streams specified to finish executing all kernels/events/barriers. Does not block any of the streamsToWaitOn. Current device only.
- `cutorch.streamWaitForMultiDevice(deviceWaiting, streamWaiting, {[device]={streamsToWaitOn...}...})`: (deviceWaiting, streamWaiting) will wait on the list of (`device`, `streams`...) pairs; handles single or multiple device. `cutorch.streamWaitForMultiDevice, a, b, {[a]={streams...}})` is equivalent to `cutorch.setDevice(a); cutorch.streamWaitFor(b, {streams...})`.
- `cutorch.streamBarrier({streams...})`: an N-to-N-way barrier between all the streams; all streams will wait for the completion of all other streams on the current device only. More efficient than creating the same N-to-N-way dependency via `streamWaitFor`.
- `cutorch.streamBarrierMultiDevice({[device]={streamsToWaitOn...}...})`: As with streamBarrier but allows barriers between streams on arbitrary devices. Creates a cross-device N-to-N-way barrier between all (device, stream) values listed.
- `cutorch.streamSynchronize(stream)`: equivalent to `cudaStreamSynchronize(stream)` for the current device. Blocks the CPU until stream completes its queued kernels/events.
