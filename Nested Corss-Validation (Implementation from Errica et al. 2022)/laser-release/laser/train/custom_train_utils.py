
def freeze_attention_weights(model, epoch):
    if epoch < 10: 
        for name, param in model.named_parameters():
            if "attention_activations" in name:
                param.requires_grad = False
    else: 
        for name, param in model.named_parameters():
            if "attention_activations" in name:
                param.requires_grad = True