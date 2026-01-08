import torch

# האם CUDA זמין? (צריך לצאת True)
print(f"CUDA Available: {torch.cuda.is_available()}")

# אם כן, מה שם הכרטיס? (צריך לצאת GeForce RTX 3050...)
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: Running on CPU only")