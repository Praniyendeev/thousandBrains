# thousandBrains
An experimental model architecture inspired by the thousand brains theory by Jeff Hawkins.


- units are recursive neuron replacements in a ANN, that condition the sparsity similar to heads int MHA.
- can leakines be intoduced horizontally across units to allow for passing of information?
- is this even theoretical different from the sparse network used by numenta on MNIST


- RL oriented framework can the weight updates from loss be completely updated in terms of reward for only "fired" neurons.
- Probably requires a defintion for fired for each neuron? Would this have an effect similar to L1 loss?
- ANNs infact only update based on when the target is missed. If the prediction is right, the loss is essentially zero and not a significant update without momentum terms. However, in this case the weights have to be strengthened. Will a simple strategy of adding an additional term proportional to activation work? LOL this is RELU. Sigmoid and tanh try to emulate firing behvaiour but relu perform better because the updates are allowed to be proportional to the input, which impacts teh update as well proportionally, thus strengthening also proportionally?
You want stengthening of all input to a neuron if they are all recieved at same time, no concept of time.


- timing is crucial in BNNs and its been proven that neurons use spiking frequency and timings to encode continuous signals.
- Can timing be modeled without explicitly using timing, like the transition from rnn to transformer
- How do self feedback loops/control signals fit into this? Mamba and TransformerFAM. 

- Can GNN type architectures help in modelling time steps and emulate propogation of data between neurons?
- Could you introduce learnable delay parameters between neurons, for weighting the message passing information.

- Recent Yolov9 uses a an auxilary branch to propogate more information rich gradients, can multiple auxilary branches improve propogation flow?
- learnable normalization, has an overseeing neuron for each layer that can be used as auxiallry back bone as well as to control short term activations based on longterm knowledge. Essentially like batch or layer norm. Learns during training, conditions based on all seen data till now.
- LIKE in teacher student models, a slow but same architecture essentially is used to generate targets (pseudo self predictions).


- Is masked prediction the optimal auxillary/pseudo task for learning multimodal information.

- for i in range(rec):x=block(x)::essentially RNN
