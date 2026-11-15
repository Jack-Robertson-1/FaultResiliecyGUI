import torch
from decode import decode_output
from models import streams
from methods.common import vote, detections_agree

def dwc_inference_serial(img, model_list):
    """Run 2 models, only run a 3rd if they disagree."""
    with torch.no_grad():
        raw1 = model_list[0](img)
        r1 = decode_output(raw1, model_list[0])
        raw2 = model_list[1](img)
        r2 = decode_output(raw2, model_list[1])

    if detections_agree(r1, r2):
        return r1, 2

    with torch.no_grad():
        raw3 = model_list[2](img)
        r3 = decode_output(raw3, model_list[2])
    return vote([r1, r2, r3]), 3

def dwc_inference_parallel(img, model_list):
    """Same as dwc_inference_serial, but the first 2 runs are on separate streams."""
    raw_results = [None, None]
    for i in range(2):
        with torch.cuda.stream(streams[i]):
            with torch.no_grad():
                raw_results[i] = model_list[i](img)
    torch.cuda.synchronize()

    r1 = decode_output(raw_results[0], model_list[0])
    r2 = decode_output(raw_results[1], model_list[1])

    if detections_agree(r1, r2):
        return r1, 2

    with torch.no_grad():
        raw3 = model_list[2](img)
        r3 = decode_output(raw3, model_list[2])
    return vote([r1, r2, r3]), 3
