import torch.nn as nn

ARCHITECTURES = {
    #modèle avec 4 couches cachées de 128 neurones
    "mlp_128_128_128_128": [128, 128, 128, 128],

    #modèle avec 3 couches
    "mlp_128_128_64": [128, 128, 64],

    #modèle à largeur décroissante avec 4 couche
    "mlp_128_128_64_32": [128, 128, 64, 32],

    #même architecture que le deuxième avec contraintes dans la loss
    "mlp_128_128_64_contraint": [128, 128, 64],
}

class VGNeuralNetwork(nn.Module):
    """Multi-Layer Perceptron qui apprend à prédire le prix d'un call à partir de (T, K, kappa, theta, sigma)
    les paramètres du modèle Variance Gamma."""

    def __init__(self, input_size=5, hidden_sizes=None, output_size=1):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = ARCHITECTURES["mlp_128_128_64"]
        layers = []
        previous = input_size
        for width in hidden_sizes:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, output_size))  # sortie linéaire
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
