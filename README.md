# thousandBrains

An experimental model architecture inspired by the Thousand Brains Theory by Jeff Hawkins.

Concepts and Ideas
1. Recursive Neuron Replacements in ANN\\
Units are recursive neuron replacements in an artificial neural network (ANN) that condition the sparsity similar to heads in multi-head attention (MHA).
Introducing leakiness horizontally across units to allow for the passing of information.
Investigate if this approach is theoretically different from the sparse network used by Numenta on MNIST.
2. Reinforcement Learning Oriented Framework
Explore if weight updates from loss can be completely updated in terms of reward for only "fired" neurons.
This likely requires a definition for "fired" for each neuron.
Would this have an effect similar to L1 loss?
ANNs only update when the target is missed. If the prediction is correct, the loss is essentially zero, leading to no significant update without momentum terms. However, in this case, the weights need to be strengthened.
A simple strategy of adding an additional term proportional to activation may work, similar to ReLU.
Sigmoid and tanh functions try to emulate firing behavior, but ReLU performs better because the updates are allowed to be proportional to the input, thus strengthening proportionally as well.
Aim for strengthening of all inputs to a neuron if they are received simultaneously, eliminating the concept of time.
3. Timing in Biological Neural Networks
Timing is crucial in biological neural networks, where neurons use spiking frequency and timings to encode continuous signals.
Can timing be modeled without explicitly using timing, like the transition from RNN to transformer architectures?
How do self-feedback loops/control signals fit into this (e.g., Mamba and TransformerFAM)?
4. Graph Neural Network Architectures
Explore if GNN-type architectures can help in modeling time steps and emulating the propagation of data between neurons.
Introduce learnable delay parameters between neurons to weight the message-passing information.
5. Auxiliary Branches in YOLOv9
YOLOv9 uses an auxiliary branch to propagate more information-rich gradients. Investigate if multiple auxiliary branches can improve propagation flow.
6. Learnable Normalization
Implement learnable normalization with an overseeing neuron for each layer that acts as an auxiliary backbone and controls short-term activations based on long-term knowledge.
This neuron learns during training and conditions based on all seen data thus far.
Similar to batch or layer normalization.
In teacher-student models, a slow but identical architecture generates targets (pseudo self-predictions).
7. Masked Prediction
Determine if masked prediction is the optimal auxiliary/pseudo task for learning multimodal information.
8. Recurrent Block Structure
Implement a structure where for i in range(rec): x=block(x), essentially forming an RNN.
Implementation
Organize and structure the above concepts into a cohesive project that can be shared on GitHub. Ensure each idea is well-documented with the necessary theoretical background, implementation details, and potential experimental setups.
9. Complex valued Networks, Physics Inspired Networks
As mentioned in 3. Timing is crucial in BNNs, Can complex networks model the phase information hence timing without specifically requiring a RNN like approach. 

Implementation:

1. Define a basic recursive logic unit, that has sparsity and WTA(winner Takes All) Mechanisms at all levels.
