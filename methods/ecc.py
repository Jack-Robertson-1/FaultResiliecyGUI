import random
from faults import fault_fns_with_count, flip_bits

# parity bits needed per 32-bit weight (SEC ~log2(32), SECDED = SEC+1, DEC ~2x SEC)
ECC_BITS = {"SEC": 6, "SECDED": 7, "DEC": 13}

def compute_ecc_overhead(model):
    """Print storage overhead for each ECC type, based on the model's real parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    base_mb = total_params * 32 / 8 / 1024 / 1024
    print(f"\n{total_params:,} params, ~{base_mb:.1f} MB base weight storage")
    for name, bits in ECC_BITS.items():
        extra_mb = total_params * bits / 8 / 1024 / 1024
        pct = bits / 32 * 100
        print(f"  {name}: +{bits} bits/weight, {pct:.1f}% overhead, ~{extra_mb:.1f} MB extra")

def inject_weight_fault(model, fault_type="single"):
    """Pick a random weight and corrupt it. diff is the exact number of bits
    that ended up different, tracked at corruption time - not re-measured
    from the resulting float values, since NaN results don't reliably keep
    their exact bit pattern across pack/unpack and would throw the count off."""
    target_params = [p for name, p in model.named_parameters() if "weight" in name]
    param = random.choice(target_params)
    flat = param.data.view(-1)
    idx = random.randint(0, flat.numel() - 1)
    orig_val = flat[idx].item()
    corrupted_val, diff = fault_fns_with_count[fault_type](orig_val)
    return param, idx, orig_val, corrupted_val, diff

def apply_sec(orig_val, corrupted_val, diff):
    """SEC: corrects 1 bit, miscorrects 2 bits (fixes the wrong bit), misses 3+."""
    if diff <= 1:
        return orig_val, "corrected"
    if diff == 2:
        return flip_bits(corrupted_val, 1), "miscorrected"
    return corrupted_val, "undetected"

def apply_secded(orig_val, corrupted_val, diff):
    """SECDED: corrects 1 bit, detects-but-can't-correct 2 bits, misses 3+."""
    if diff <= 1:
        return orig_val, "corrected"
    if diff == 2:
        return corrupted_val, "detected_uncorrected"
    return corrupted_val, "undetected"

def apply_dec(orig_val, corrupted_val, diff):
    """DEC: corrects 1 or 2 bits, misses 3+."""
    if diff <= 2:
        return orig_val, "corrected"
    return corrupted_val, "undetected"

ECC_VARIANTS = {"SEC": apply_sec, "SECDED": apply_secded, "DEC": apply_dec}

def restore_weight(param, idx, orig_val):
    """Put the original weight value back after a trial."""
    flat = param.data.view(-1)
    flat[idx] = orig_val