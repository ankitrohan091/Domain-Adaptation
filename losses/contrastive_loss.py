# losses/contrastive_loss.py
import torch
import torch.nn.functional as F

def contrastive_loss(prototypes, instance_embeddings, pseudo_labels, temperature=0.05):
    """
    Multi-prototype contrastive loss used in DA-RAW.
    Args:
        prototypes: dict of class_id -> [K, D] learnable prototype tensor
        instance_embeddings: [N, D] tensor of projected instance features
        pseudo_labels: [N] tensor of class ids (including background)
        temperature: softmax temperature
    Returns:
        loss: scalar contrastive loss
    """
    total_loss = 0.0
    num_instances = instance_embeddings.size(0)

    for i in range(num_instances):
        z = instance_embeddings[i]  # [D]
        class_id = pseudo_labels[i].item()

        if class_id not in prototypes:
            continue

        class_prototypes = prototypes[class_id]  # [K, D]

        # Compute cosine similarities with all class prototypes
        sims = F.cosine_similarity(z.unsqueeze(0), class_prototypes, dim=1) / temperature  # [K]
        probs = F.softmax(sims, dim=0)
        log_probs = torch.log(probs + 1e-8)  # for numerical stability

        # Uniform soft target (equipartition assumption)
        uniform_target = torch.full_like(probs, 1.0 / len(probs))
        loss = F.kl_div(log_probs, uniform_target, reduction='batchmean')
        total_loss += loss

    return total_loss / num_instances
