import os
import time
import json
import platform
import torch

from models import models, wrapper_a
from faults import inject_fault_direct
from decode import load_image_tensor, decode_output
from methods.common import vote, classify
from methods.tmr import serial_inference, parallel_inference
from methods.dwc import dwc_inference_serial, dwc_inference_parallel
from methods.ecc import compute_ecc_overhead, inject_weight_fault, restore_weight, ECC_VARIANTS

CHECKPOINT_FILE = "checkpoint_results.json"
FRESH_START = True  # False to resume instead of wiping checkpoint
N_TRIALS = 10

# which methods to actually run
RUN_TMR = False
RUN_DWC = False
RUN_ECC = True

if FRESH_START and os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("FRESH_START=True — deleted old checkpoint, starting clean.")

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f:
        results_summary = json.load(f)
    print(f"Resuming from checkpoint — {sum(len(v) for v in results_summary.values())} combos already done")
else:
    results_summary = {}

if platform.system() == "Windows":
    test_dir = r"C:\Users\218ro\rareplanes\test\PS-RGB_tiled"
else:
    test_dir = "/home/shrec/test_images"
files = os.listdir(test_dir)

# images used over the summer
old_image_names = [
    "10_104001002E88FE00_tile_10",
    "35_10400100363F0700_tile_112",
    "108_10400100457A8A00_tile_68",
    "62_104001003CA27B00_tile_82",
    "67_1040010046D15B00_tile_1627",
    "96_104001003C8C7800_tile_1050",
    "93_104001002A4FAA00_tile_1366",
    "84_1040010006ABC200_tile_111",
    "67_104001003F0B3200_tile_1392",
    "106_104001003D8DB300_tile_87",
]

selected_files = []
for name in old_image_names:
    match = [f for f in files if name in f]
    if match:
        selected_files.append(match[0])
    else:
        print(f"WARNING: no match found for {name}")

print("Using images:", selected_files)
test_images = [os.path.join(test_dir, f) for f in selected_files]

# ---------- EXPERIMENTATION ----------

print("\nComputing golden baselines...")
golden = {}
for path in test_images:
    out = wrapper_a.predict(path, conf=0.25, verbose=False)
    golden[path] = [(int(b.cls), float(b.conf)) for b in out[0].boxes]
    print(f"  {os.path.basename(path)}: {len(golden[path])} detections")

# standalone sanity check - run this separately, not mixed into the main loop
for path in test_images:
    img_tensor = load_image_tensor(path)
    with torch.no_grad():
        raw = models[0](img_tensor)
    clean_result = decode_output(raw, models[0])
    g = golden[path]
    print(os.path.basename(path))
    print("  golden:", sorted([c[0] for c in g]))
    print("  clean: ", sorted([c[0] for c in clean_result]))
    print("  match:", sorted([c[0] for c in g]) == sorted([c[0] for c in clean_result]))

if RUN_ECC:
    compute_ecc_overhead(models[0])

fault_types = ["single", "multi", "stuck"]
run_configs = {"1_of_3": [0], "2_of_3": [0, 1], "3_of_3": [0, 1, 2]}

# ecc_stats keyed by fault_type -> variant -> counts, so we can break it down later
ecc_stats = {
    ft: {name: {"sdc": 0, "masked": 0, "miscorrected": 0,
                "detected_uncorrected": 0, "undetected": 0, "times": []}
         for name in ECC_VARIANTS}
    for ft in fault_types
}

print("\nRunning experiment...")
for fault_type in fault_types:
    for config_name, faulted_idx in run_configs.items():

        if config_name in results_summary.get(fault_type, {}):
            print(f"Skipping {fault_type} / {config_name} (already done)")
            continue

        sdc_count_s, masked_count_s = 0, 0
        sdc_count_p, masked_count_p = 0, 0
        dwc_sdc_count_s, dwc_masked_count_s = 0, 0
        dwc_sdc_count_p, dwc_masked_count_p = 0, 0
        parallel_times, serial_times = [], []
        dwc_serial_times, dwc_parallel_times = [], []
        dwc_runs_s, dwc_runs_p = [], []

        for path in test_images:
            g = golden[path]
            img_tensor = load_image_tensor(path)

            for _ in range(N_TRIALS):
                handles = []
                if RUN_TMR or RUN_DWC:
                    for idx in faulted_idx:
                        h = inject_fault_direct(models[idx], fault_type=fault_type)
                        handles.append(h)

                if RUN_TMR:
                    t0 = time.time()
                    s_results = serial_inference(img_tensor, models)
                    serial_times.append((time.time() - t0) * 1000)

                    t0 = time.time()
                    p_results = parallel_inference(img_tensor, models)
                    parallel_times.append((time.time() - t0) * 1000)

                    outcome_s = classify(g, vote(s_results))
                    if outcome_s == "SDC":
                        sdc_count_s += 1
                    else:
                        masked_count_s += 1

                    outcome_p = classify(g, vote(p_results))
                    if outcome_p == "SDC":
                        sdc_count_p += 1
                    else:
                        masked_count_p += 1

                if RUN_DWC:
                    t0 = time.time()
                    dwc_r_s, runs_s = dwc_inference_serial(img_tensor, models)
                    dwc_serial_times.append((time.time() - t0) * 1000)
                    dwc_runs_s.append(runs_s)

                    t0 = time.time()
                    dwc_r_p, runs_p = dwc_inference_parallel(img_tensor, models)
                    dwc_parallel_times.append((time.time() - t0) * 1000)
                    dwc_runs_p.append(runs_p)

                    dwc_outcome_s = classify(g, dwc_r_s)
                    if dwc_outcome_s == "SDC":
                        dwc_sdc_count_s += 1
                    else:
                        dwc_masked_count_s += 1

                    dwc_outcome_p = classify(g, dwc_r_p)
                    if dwc_outcome_p == "SDC":
                        dwc_sdc_count_p += 1
                    else:
                        dwc_masked_count_p += 1

                if RUN_ECC:
                    # single model, no redundancy, same injected fault
                    # tested against all 3 variants so it's a fair comparison
                    param, w_idx, orig_val, corrupted_val, diff = inject_weight_fault(models[0], fault_type=fault_type)

                    for variant_name, apply_fn in ECC_VARIANTS.items():
                        t0 = time.time()
                        final_val, status = apply_fn(orig_val, corrupted_val, diff)
                        param.data.view(-1)[w_idx] = final_val
                        with torch.no_grad():
                            raw = models[0](img_tensor)
                        ecc_result = decode_output(raw, models[0])
                        restore_weight(param, w_idx, orig_val)
                        ecc_stats[fault_type][variant_name]["times"].append((time.time() - t0) * 1000)

                        if status in ("miscorrected", "detected_uncorrected", "undetected"):
                            ecc_stats[fault_type][variant_name][status] += 1

                        ecc_outcome = classify(g, ecc_result)
                        if ecc_outcome == "SDC":
                            ecc_stats[fault_type][variant_name]["sdc"] += 1
                        else:
                            ecc_stats[fault_type][variant_name]["masked"] += 1

                for h in handles:
                    h.remove()

        if RUN_TMR:
            total_s = sdc_count_s + masked_count_s
            sdc_pct_s = (sdc_count_s / total_s * 100) if total_s else 0
            total_p = sdc_count_p + masked_count_p
            sdc_pct_p = (sdc_count_p / total_p * 100) if total_p else 0
            avg_p = sum(parallel_times) / len(parallel_times)
            avg_s = sum(serial_times) / len(serial_times)
        else:
            sdc_pct_s = sdc_pct_p = avg_p = avg_s = 0

        if RUN_DWC:
            dwc_total_s = dwc_sdc_count_s + dwc_masked_count_s
            dwc_sdc_pct_s = (dwc_sdc_count_s / dwc_total_s * 100) if dwc_total_s else 0
            dwc_total_p = dwc_sdc_count_p + dwc_masked_count_p
            dwc_sdc_pct_p = (dwc_sdc_count_p / dwc_total_p * 100) if dwc_total_p else 0
            avg_dwc_s = sum(dwc_serial_times) / len(dwc_serial_times)
            avg_dwc_p = sum(dwc_parallel_times) / len(dwc_parallel_times)
            avg_runs_s = sum(dwc_runs_s) / len(dwc_runs_s)
            avg_runs_p = sum(dwc_runs_p) / len(dwc_runs_p)
        else:
            dwc_sdc_pct_s = dwc_sdc_pct_p = avg_dwc_s = avg_dwc_p = avg_runs_s = avg_runs_p = 0

        results_summary.setdefault(fault_type, {})[config_name] = {
            "sdc_pct_s": sdc_pct_s, "sdc_pct_p": sdc_pct_p, "avg_p": avg_p, "avg_s": avg_s,
            "dwc_sdc_pct_s": dwc_sdc_pct_s, "dwc_sdc_pct_p": dwc_sdc_pct_p,
            "avg_dwc_s": avg_dwc_s, "avg_dwc_p": avg_dwc_p,
            "avg_runs_s": avg_runs_s, "avg_runs_p": avg_runs_p
        }

        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"  Checkpoint saved ({fault_type}/{config_name})")

# golden baseline summary, printed before results so the tables have context
print("\n" + "=" * 70)
print("GOLDEN BASELINE")
print("=" * 70)
print(f"{'Image':<50} {'Detections':>10}")
print("-" * 70)
for path in test_images:
    print(f"{os.path.basename(path):<50} {len(golden[path]):>10}")

if RUN_TMR:
    print("\n" + "=" * 70)
    print("TMR RESULTS")
    print("=" * 70)
    print(f"{'Fault':<8} {'Config':<8} {'SDC% (par)':>11} {'SDC% (ser)':>11} {'Time (par)':>11} {'Time (ser)':>11}")
    print("-" * 70)
    for ft in fault_types:
        for cfg in run_configs:
            d = results_summary[ft][cfg]
            print(f"{ft:<8} {cfg:<8} {d['sdc_pct_p']:>10.1f}% {d['sdc_pct_s']:>10.1f}% {d['avg_p']:>9.1f}ms {d['avg_s']:>9.1f}ms")

if RUN_DWC:
    print("\n" + "=" * 70)
    print("DWC w/ RETRY+VOTE RESULTS")
    print("=" * 70)
    print(f"{'Fault':<8} {'Config':<8} {'SDC% (par)':>11} {'SDC% (ser)':>11} {'Time (par)':>11} {'Time (ser)':>11} {'Runs (par)':>11} {'Runs (ser)':>11}")
    print("-" * 70)
    for ft in fault_types:
        for cfg in run_configs:
            d = results_summary[ft][cfg]
            print(f"{ft:<8} {cfg:<8} {d['dwc_sdc_pct_p']:>10.1f}% {d['dwc_sdc_pct_s']:>10.1f}% {d['avg_dwc_p']:>9.1f}ms {d['avg_dwc_s']:>9.1f}ms {d['avg_runs_p']:>10.2f} {d['avg_runs_s']:>10.2f}")

if RUN_ECC:
    print("\n" + "=" * 70)
    print("ECC RESULTS (by fault type)")
    print("=" * 70)
    print(f"{'Fault':<8} {'Variant':<8} {'SDC%':>8} {'Miscorr':>8} {'Det-NC':>8} {'Undet':>8} {'Avg ms':>8}")
    print("-" * 70)
    for ft in fault_types:
        for name in ECC_VARIANTS:
            s = ecc_stats[ft][name]
            total = s["sdc"] + s["masked"]
            sdc_pct = (s["sdc"] / total * 100) if total else 0
            avg_t = sum(s["times"]) / len(s["times"]) if s["times"] else 0
            print(f"{ft:<8} {name:<8} {sdc_pct:>7.1f}% {s['miscorrected']:>8} {s['detected_uncorrected']:>8} {s['undetected']:>8} {avg_t:>7.2f}")
        print("-" * 70)

print("\n" + "=" * 70)

