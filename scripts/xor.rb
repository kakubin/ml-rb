require 'torch-rb'

data = [
  [[0, 0], 0],
  [[1, 1], 0],
  [[1, 0], 1],
  [[0, 1], 1],
].map { |input, target| [Torch::Tensor.new(input), Torch::Tensor.new([target])] }

class MyNet < Torch::NN::Module
  def initialize
    super
    @linear1 = Torch::NN::Linear.new(2, 8)
    @linear2 = Torch::NN::Linear.new(8, 1)
  end

  def forward(x)
    x = Torch::NN::F.relu(@linear1.call(x))
    @linear2.call(x)
  end
end

net = MyNet.new
criterion = Torch::NN::MSELoss.new
optimizer = Torch::Optim::Adam.new(net.parameters, lr: 0.01)

1000.times do
  data.each do |input, target|
    output = net.call(input)
    optimizer.zero_grad()
    loss = criterion.call(output, target)
    puts loss
    loss.backward
    optimizer.step
  end
end

puts net.call(Torch::Tensor.new([1, 0]))
puts net.call(Torch::Tensor.new([0, 1]))
puts net.call(Torch::Tensor.new([0, 0]))
puts net.call(Torch::Tensor.new([1, 1]))
