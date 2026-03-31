import torch
import torch.nn as nn

try:
    print("Trying to load with weights_only=False...")
    model_data = torch.load('diabetic_retinopathy_full_model.pth', map_location='cpu', weights_only=False)
    print(f"Loaded successfully. Type: {type(model_data)}")
    
    if hasattr(model_data, "fc"):
        print(f"FC LAYER: {model_data.fc}")
    elif hasattr(model_data, "classifier"):
        print(f"CLASSIFIER: {model_data.classifier}")
    else:
        print("No standard FC or classifier head found.")

except Exception as e:
    print(f"Failed: {e}")
