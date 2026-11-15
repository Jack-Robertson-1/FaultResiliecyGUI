import torch
from decode import decode_output
from models import streams

def run_on_stream(model, img, stream, results, idx):
    """Run one model on its own stream and decode immediately. Not used in the main loop - see parallel_inference."""
    with torch.cuda.stream(stream):
        with torch.no_grad():
            raw = model(img)
            results[idx] = decode_output(raw, model)

def serial_inference(img, model_list):
    """Run all 3 models one after another."""
    results = [None, None, None]
    for i, m in enumerate(model_list):
        with torch.no_grad():
            raw = m(img)
            results[i] = decode_output(raw, m)
    torch.cuda.synchronize()
    return results

def parallel_inference(img, model_list):
    """Launch all 3 forward passes on separate streams, sync once, then decode.
    Decoding before the sync blocks each stream in turn and kills the overlap."""
    raw_results = [None, None, None]
    for i, (m, s) in enumerate(zip(model_list, streams)):
        with torch.cuda.stream(s):
            with torch.no_grad():
                raw_results[i] = m(img)
    torch.cuda.synchronize()

    results = [decode_output(raw_results[i], model_list[i]) for i in range(3)]
    return results
