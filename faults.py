import torch
import random, struct

def flip_bits(val, num_bits):
    """Flip num_bits random bits in a float32's raw bit pattern."""
    b = struct.pack('f', val)
    i = struct.unpack('I', b)[0]
    for _ in range(num_bits):
        i ^= (1 << random.randint(0, 31))
    return struct.unpack('f', struct.pack('I', i))[0]

def flip_bits_count(val, num_bits):
    """Same as flip_bits, but also returns the exact number of bits that ended
    up different (as a pure integer XOR popcount, computed before the result
    ever goes through a float round-trip). Needed because NaN values don't
    reliably keep their exact bit pattern across struct pack/unpack, so
    re-measuring the diff later from the float values can be off by a bit
    or two whenever a flip happens to land in the exponent and produce a NaN."""
    b = struct.pack('f', val)
    i = struct.unpack('I', b)[0]
    mask = 0
    for _ in range(num_bits):
        mask ^= (1 << random.randint(0, 31))
    corrupted_val = struct.unpack('f', struct.pack('I', i ^ mask))[0]
    return corrupted_val, bin(mask).count('1')

def single_bit(val):
    """Flip exactly 1 bit."""
    return flip_bits(val, 1)

def multi_bit(val):
    """Flip 2-7 bits at once."""
    return flip_bits(val, random.randint(2, 7))

def stuck_bit(val):
    """Force one random bit permanently to 0 or 1."""
    b = struct.pack('f', val)
    i = struct.unpack('I', b)[0]
    bit = random.randint(0, 31)
    if random.randint(0, 1):
        i |= (1 << bit)
    else:
        i &= ~(1 << bit)
    return struct.unpack('f', struct.pack('I', i))[0]

def stuck_bit_count(val):
    """Same as stuck_bit, but also returns 0 or 1 - the actual bit changed,
    not re-measured from the float (see flip_bits_count for why)."""
    b = struct.pack('f', val)
    i = struct.unpack('I', b)[0]
    bit = random.randint(0, 31)
    old_bit = (i >> bit) & 1
    new_bit = random.randint(0, 1)
    if new_bit:
        i |= (1 << bit)
    else:
        i &= ~(1 << bit)
    corrupted_val = struct.unpack('f', struct.pack('I', i))[0]
    return corrupted_val, 0 if old_bit == new_bit else 1

fault_fns = {"single": single_bit, "multi": multi_bit, "stuck": stuck_bit}
fault_fns_with_count = {
    "single": lambda val: flip_bits_count(val, 1),
    "multi": lambda val: flip_bits_count(val, random.randint(2, 7)),
    "stuck": stuck_bit_count,
}

def inject_fault_direct(model, fault_type="single"):
    """Hook a random conv layer's output, corrupt one value. Returns a removable handle."""
    target_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    layer_idx = random.randint(0, len(target_layers) - 1)
    target_layer = target_layers[layer_idx]
    fault_fn = fault_fns[fault_type]

    def hook(module, input_val, output):
        _, C, H, W = output.shape
        c = random.randint(0, C - 1)
        h = random.randint(0, H - 1)
        w = random.randint(0, W - 1)
        orig_val = output[0, c, h, w].item()
        output[0, c, h, w] = fault_fn(orig_val)
        return output

    handle = target_layer.register_forward_hook(hook)
    return handle