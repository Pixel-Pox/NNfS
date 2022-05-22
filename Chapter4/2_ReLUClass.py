import numpy as np

class Activation_ReLU:
    def forward(self, inputs) -> None:
        self.output = np.maximum(0, inputs)
