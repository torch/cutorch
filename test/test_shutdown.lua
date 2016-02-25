local Threads = require 'threads'
require 'cutorch'

print('Memory usage before intialization of threads [free memory], [total memory]')
print(cutorch.getMemoryUsage())
threads = Threads(100, function() require 'cutorch' end)
print('Memory usage after intialization of threads [free memory], [total memory]')
print(cutorch.getMemoryUsage())
threads:terminate()
print('Memory usage after termination of threads [free memory], [total memory]')
print(cutorch.getMemoryUsage())
