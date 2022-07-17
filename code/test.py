


x = input('Which DS to test 1: MNIST, 2: CIFAR, 3: INTEL')


if x == 0:
    from test_mnist import *
elif x == 1:
    from test_cif import *
elif x == 2:
    from test_int import *







#accuracy_loss(val=False)
#accuracy_loss_batch()
#gradients()
#activations()
#feature_map()
#activations_input()
#feature_map_input()
#gradients_input()
#feature_map_per_layer()
#feature_map_grad()
#gradients_output()
#gradients_hook_classes()
#gradients_input_output()
#gradients_input_output_all_layers()