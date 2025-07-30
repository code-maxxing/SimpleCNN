CIFAR-10 Classifier: My First PyTorch CNN 

Built a convolutional neural network using PyTorch to classify CIFAR-10 images.
Implemented a full training pipeline with GPU support, data normalization (transforms.Normalize), and cross-entropy loss. 
The architecture uses two convolutional layers (nn.Conv2d) with ReLU activation and max-pooling, followed by three linear layers -
The training loop handles DataLoader (as batch iteration) and backpropagation (loss.backward()) to learn.
Validation metrics track accuracy and loss per epoch. 

heres the documented result after a small test of 5 epochs:

Epoch 1: Train Loss 1.489  Val Acc 55.73%  
Epoch 2: Train Loss 1.123  Val Acc 62.71%  
Epoch 3: Train Loss 0.962  Val Acc 66.12%  
Epoch 4: Train Loss 0.857  Val Acc 67.76%  
Epoch 5: Train Loss 0.779  Val Acc 67.00%

Peak validation accuracy was hit at epoch 4, at 67.76%, not outstanding by far but functional for a first time clone.

Started by cloning patrick loebar's examples and basic cnns i found on stack overflow,
using ai assistants to help debug and fix code, implementing it after understanding and cross referencing with tutorials

Looked through a few code reviews and understood how far below bar my current work was, I aim to add significantly higher (till distinguished) accuracy validations by vertically building on this consistently
