import torch
from ultralytics import YOLO
from pytorchfi.core import fault_injection
import copy

device = torch.device("cuda")

# 3 model copies for TMR/DWC
base = YOLO("best.pt")
model_a = base.model.to(device).eval()
model_b = copy.deepcopy(model_a)
model_c = copy.deepcopy(model_a)
models = [model_a, model_b, model_c]

# wrapper used only for golden baseline (.predict() is fine here, speed doesn't matter)
wrapper_a = YOLO("best.pt")
wrapper_a.model = model_a

# PyTorchFI wrappers - not used in main loop, kept for reference
pfi_models = [
    fault_injection(m, batch_size=1,
                    input_shape=[3, 640, 640],
                    layer_types=[torch.nn.Conv2d],
                    use_cuda=True)
    for m in models
]

# one CUDA stream per model
streams = [torch.cuda.Stream() for _ in range(3)]
