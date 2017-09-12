local tester = torch.Tester()

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Testing alias multinomial on cuda')
cmd:text()
cmd:text('Options')
cmd:option('--compare',false,'compare with cutorch multinomial')
cmd:text()

-- parse input params
params = cmd:parse(arg)

require 'cutorch'
local function checkMultinomial()
   local n_class = {10, 100, 1000}
   local n_sample = {10, 100, 1000, 10000}
   local n_dist = 100
   for _, curr_n_class in pairs(n_class) do
      for _, curr_n_sample in pairs(n_sample) do
	 print("")
	 print("Benchmarking multinomial with "..curr_n_class.." classes and "..curr_n_sample.." samples")
	 torch.seed()
	 local probs = torch.CudaDoubleTensor(n_dist, curr_n_class):uniform(0,1)
	 local a = torch.Timer()
	 local cold_time = a:time().real
	 a:reset()
	 cutorch.synchronize()
	 a:reset()
	 for i = 1,10 do
	    torch.multinomial(probs, curr_n_sample, true)
	    cutorch.synchronize()
	 end
	 print("[CUDA] : torch.multinomial draw: "..(a:time().real/10).." seconds (hot)")
      end
      torch.seed()
      local probs = torch.CudaDoubleTensor(3, curr_n_class):uniform(0,1)
      for i =1,3 do
	 probs[i]:div(probs[i]:sum())
      end
      local output = torch.multinomial(probs, 5000000, true)
      local counts = torch.Tensor(3, curr_n_class):zero()
      for i=1,3 do
	 output[i]:long():apply(function(x) counts[{i, x}] = counts[{i, x}] + 1 end)
	 counts[i]:div(counts[i]:sum())
      end
      tester:eq(probs:double(), counts, 0.01, "probs and counts should be approximately equal for n_class = "..curr_n_class)
   end
end
tester:add(checkMultinomial)
tester:run()
