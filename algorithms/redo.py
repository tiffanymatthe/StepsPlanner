import torch
from torch import optim
import torch.nn as nn

@torch.no_grad()
def reset_adam_moments(optimizer: optim.Adam, reset_masks: dict[str, torch.Tensor]) -> optim.Adam:
    """Resets the moments of the Adam optimizer for the dormant neurons."""

    assert isinstance(optimizer, (optim.Adam, optim.AdamW)), "Moment resetting currently only supported for Adam optimizer"
    for i, mask in enumerate(reset_masks):
        # Reset the moments for the weights
        optimizer.state_dict()["state"][i * 2]["exp_avg"][mask, ...] = 0.0
        optimizer.state_dict()["state"][i * 2]["exp_avg_sq"][mask, ...] = 0.0
        # NOTE: Step count resets are key to the algorithm's performance
        # It's possible to just reset the step for moment that's being reset
        optimizer.state_dict()["state"][i * 2]["step"].zero_()

        # Reset the moments for the bias
        optimizer.state_dict()["state"][i * 2 + 1]["exp_avg"][mask] = 0.0
        optimizer.state_dict()["state"][i * 2 + 1]["exp_avg_sq"][mask] = 0.0
        optimizer.state_dict()["state"][i * 2 + 1]["step"].zero_()

        # Reset the moments for the output weights
        if (
            len(optimizer.state_dict()["state"][i * 2]["exp_avg"].shape) == 4
            and len(optimizer.state_dict()["state"][i * 2 + 2]["exp_avg"].shape) == 2
        ):
            # Catch transition from conv to linear layer through moment shapes
            num_repeatition = optimizer.state_dict()["state"][i * 2 + 2]["exp_avg"].shape[1] // mask.shape[0]
            linear_mask = torch.repeat_interleave(mask, num_repeatition)
            optimizer.state_dict()["state"][i * 2 + 2]["exp_avg"][:, linear_mask] = 0.0
            optimizer.state_dict()["state"][i * 2 + 2]["exp_avg_sq"][:, linear_mask] = 0.0
            optimizer.state_dict()["state"][i * 2 + 2]["step"].zero_()
        else:
            # Standard case: layer and next_layer are both conv or both linear
            # Reset the outgoing weights to 0
            optimizer.state_dict()["state"][i * 2 + 2]["exp_avg"][:, mask, ...] = 0.0
            optimizer.state_dict()["state"][i * 2 + 2]["exp_avg_sq"][:, mask, ...] = 0.0
            optimizer.state_dict()["state"][i * 2 + 2]["step"].zero_()

    return optimizer

@torch.inference_mode()
def get_redo_masks(activations: list[torch.Tensor], tau: float) -> torch.Tensor:
    """
    Computes the ReDo mask for a given set of activations.
    The returned mask has True where neurons are dormant and False where they are active.
    """
    masks = []

    # Last activation are the q-values, which are never reset
    for activation in activations:
        # Taking the mean here conforms to the expectation under D in the main paper's formula
        score = activation.abs().mean(dim=0)

        # Divide by activation mean to make the threshold independent of the layer size
        # see https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
        # https://github.com/google/dopamine/issues/209
        normalized_score = score / (score.mean() + 1e-9)

        layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)
        if tau > 0.0:
            layer_mask[normalized_score <= tau] = 1
        else:
            layer_mask[torch.isclose(normalized_score, torch.zeros_like(normalized_score))] = 1
        masks.append(layer_mask)
    return masks

@torch.no_grad()
def _masked_init_r_(
    layer: nn.Linear,
    mask: torch.Tensor
) -> None:
    """Partially re-initializes the weights and biases of a layer according to the init_r_ logic."""

    # Initialize the masked weights with orthogonal_ initialization
    nn.init.orthogonal_(layer.weight[mask])
    
    # Apply gain (relu) to the masked weights
    layer.weight[mask] *= nn.init.calculate_gain("relu")
    
    # Reinitialize the biases for the masked neurons to 0, if the layer has biases
    if layer.bias is not None:
        nn.init.constant_(layer.bias[mask], 0.0)

@torch.no_grad()
def reset_dormant_neurons(layers, redo_masks: torch.Tensor):
    """Re-initializes the dormant neurons of a model."""

    # layers = [(name, layer) for name, layer in list(model.named_modules())[1:]]
    # print(layers)
    assert len(redo_masks) == len(layers), "Number of masks must match the number of layers"

    # Reset the ingoing weights
    # Here the mask size always matches the layer weight size
    for i in range(len(layers)):
        mask = redo_masks[i]
        layer = layers[i]
        if i + 1 < len(layers) - 1:
            next_layer = layers[i + 1]
        else:
            next_layer = None

        # Skip if there are no dead neurons
        if torch.all(~mask):
            # No dormant neurons in this layer
            continue

        # 1. Reset the ingoing weights using the initialization distribution
        _masked_init_r_(layer, mask)

        # 2. Reset the outgoing weights to 0
        # NOTE: Don't reset the bias for the following layer or else you will create new dormant neurons
        # To not reset in the last layer: and not next_layer_name == 'q'
        # Reset the outgoing weights to 0
        if next_layer is not None:
            next_layer.weight.data[:, mask, ...] = 0.0

@torch.inference_mode()
def run_redo(
    layers_to_reset,
    activations: list[torch.Tensor],
    optimizer,
    re_initialize: bool,
    tau: float,
):
    # Masks for tau=0 logging
    zero_masks = get_redo_masks(activations, 0)
    total_neurons = sum([torch.numel(mask) for mask in zero_masks])
    zero_count = sum([torch.sum(mask) for mask in zero_masks])
    zero_fraction = (zero_count / total_neurons) * 100

    # Calculate the masks actually used for resetting
    masks = get_redo_masks(activations, tau)
    dormant_count = sum([torch.sum(mask) for mask in masks])
    dormant_fraction = (dormant_count / sum([torch.numel(mask) for mask in masks])) * 100

    # Re-initialize the dormant neurons and reset the Adam moments
    if re_initialize:
        print("Re-initializing dormant neurons")
        print(f"Total neurons: {total_neurons} | Dormant neurons: {dormant_count} | Dormant fraction: {dormant_fraction:.2f}%")
        reset_dormant_neurons(layers_to_reset, masks)
        reset_adam_moments(optimizer, masks)

    return {
        "zero_fraction": zero_fraction,
        "zero_count": zero_count,
        "dormant_fraction": dormant_fraction,
        "dormant_count": dormant_count,
    }