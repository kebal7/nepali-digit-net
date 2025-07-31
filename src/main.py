import load
import network

training_data, test_data = load.get_training_and_test_data()

net = network.Network([1024, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

net.save('digit_model.npz')
print("Model Saved!")
