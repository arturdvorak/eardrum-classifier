"""
Training Strategies Module

This module defines different fine-tuning strategies for the EfficientNetV2 model.
Each strategy freezes different parts of the model to find the best approach.
"""

def set_finetune_strategy(model, strategy):
    """
    Apply a specific fine-tuning strategy to the model.
    
    This function controls which parts of the model are trainable:
    - freeze_backbone: Only train the classifier
    - last1+head: Train last block + classifier
    - last2+head: Train last 2 blocks + classifier
    - last3+head: Train last 3 blocks + classifier
    - last4+head: Train last 4 blocks + classifier
    - full: Train the entire model
    
    Args:
        model: The EfficientNetV2 model
        strategy (str): The fine-tuning strategy to apply
    """
    
    if strategy == "freeze_backbone":
        # Freeze all backbone layers, only train classifier
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.model.get_classifier().parameters():
            param.requires_grad = True
            
    elif strategy == "last1+head":
        # Freeze all layers except last block and classifier
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.model.blocks[-1].parameters():
            param.requires_grad = True
        for param in model.model.get_classifier().parameters():
            param.requires_grad = True
            
    elif strategy == "last2+head":
        # Freeze all layers except last 2 blocks and classifier
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.model.blocks[-2:].parameters():
            param.requires_grad = True
        for param in model.model.get_classifier().parameters():
            param.requires_grad = True
            
    elif strategy == "last3+head":
        # Freeze all layers except last 3 blocks and classifier
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.model.blocks[-3:].parameters():
            param.requires_grad = True
        for param in model.model.get_classifier().parameters():
            param.requires_grad = True
            
    elif strategy == "last4+head":
        # Freeze all layers except last 4 blocks and classifier
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.model.blocks[-4:].parameters():
            param.requires_grad = True
        for param in model.model.get_classifier().parameters():
            param.requires_grad = True
            
    elif strategy == "full":
        # Train the entire model (all parameters)
        for param in model.model.parameters():
            param.requires_grad = True
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def get_available_strategies():
    """
    Get list of available fine-tuning strategies.
    
    Returns:
        list: List of strategy names
    """
    return [
        "freeze_backbone",    # Only classifier
        "last1+head",         # Last block + classifier
        "last2+head",         # Last 2 blocks + classifier
        "last3+head",         # Last 3 blocks + classifier
        "last4+head",         # Last 4 blocks + classifier
        "full"                # Entire model
    ]

if __name__ == "__main__":
    # Test the strategies
    print("Available fine-tuning strategies:")
    strategies = get_available_strategies()
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i}. {strategy}")
    
    print("\nStrategy descriptions:")
    print("  freeze_backbone: Only train the final classifier layer")
    print("  last1+head: Train the last block and classifier")
    print("  last2+head: Train the last 2 blocks and classifier")
    print("  last3+head: Train the last 3 blocks and classifier")
    print("  last4+head: Train the last 4 blocks and classifier")
    print("  full: Train the entire model from scratch")
