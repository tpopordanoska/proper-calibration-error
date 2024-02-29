import torch
from torch import nn, optim
from torch.nn import functional as F


# Modified from: https://github.com/gpleiss/temperature_scaling
class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self):
        super(ModelWithTemperature, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, logits, labels):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """

        nll_criterion = nn.CrossEntropyLoss()

        temp_values = torch.linspace(1e-4, 5, steps=10000)
        optim_temp = -1
        best_loss = torch.finfo(torch.float).max
        for temp in temp_values:
            loss = nll_criterion((logits / temp), labels)
            if loss < best_loss:
                best_loss = loss
                optim_temp = temp

        self.temperature.data = optim_temp.unsqueeze(0)

        return self

    def brier_score(self, probabilities, labels):
        # Calculate the Brier score
        return (torch.sum((probabilities - labels) ** 2, 1)).mean()

    def set_temperature_brier(self, logits, labels):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        labels_ohe = nn.functional.one_hot(labels, num_classes=logits.shape[1])

        temp_values = torch.linspace(1e-4, 5, steps=10000)
        optim_temp = -1
        best_loss = torch.finfo(torch.float).max
        for temp in temp_values:
            scaled_logits = logits / temp
            probabilities = torch.softmax(scaled_logits, dim=1)
            loss = self.brier_score(probabilities, labels_ohe)
            if loss < best_loss:
                best_loss = loss
                optim_temp = temp

        self.temperature.data = optim_temp.unsqueeze(0)

        return self
