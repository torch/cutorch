require 'cutorch'
require 'socket'

math.randomseed(1)
torch.manualSeed(1)
n_rows = 1000
n_cat = 1000
n_sample = 500
n_exp = 1
with_replace = false
x = torch.Tensor(n_exp,n_cat):uniform(0,1)
y1 = torch.multinomial(x, n_sample, with_replace)
y1 = y1:resize((n_sample))

x = torch.Tensor(n_rows, n_cat):uniform(0,1)

t1 = socket.gettime()
for i = 1,100 do
  z = x:index(2, y1)
  t2 = socket.gettime()
  print(i, t2 - t1)
end


for i = 1,100 do
  z = x:cuda():index(2, y1)
  t3 = socket.gettime()
  print(i, t3 - t2)
end
print('total time taken for c version of indexselect: ' .. t2-t1)
print('total time taken for cuda version of indexselect: ' .. t3-t2)

