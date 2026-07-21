# run_ablation_grid.py (improved)
import os, shutil, time, json, traceback, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import main experiment module (ensure this is the corrected file)
import ics_federated_experiment as base

# optionally force-this if you want to ensure the 3 datasets are used:
# base.CONFIG["datasets"] = ["MNIST", "EMNIST", "CIFAR10"]

OUTDIR = base.CONFIG.get('save_dir', "./results")
ABL_OUT = os.path.join(OUTDIR, "ablation")
os.makedirs(ABL_OUT, exist_ok=True)

# Define ablation variants (weights length 5 expected)
variants = {
    "full_ics": {"ics_weights": base.CONFIG["ics_weights"], "use_rep": base.CONFIG["use_rep"]},
    "no_ent":   {"ics_weights": [base.CONFIG["ics_weights"][0], base.CONFIG["ics_weights"][1], 0.0, base.CONFIG["ics_weights"][3], base.CONFIG["ics_weights"][4]], "use_rep": base.CONFIG["use_rep"]},
    "no_cons":  {"ics_weights": [base.CONFIG["ics_weights"][0], base.CONFIG["ics_weights"][1], base.CONFIG["ics_weights"][2], 0.0, base.CONFIG["ics_weights"][4]], "use_rep": base.CONFIG["use_rep"]},
    "no_rep":   {"ics_weights": base.CONFIG["ics_weights"][:4] + [0.0], "use_rep": False},
    "global_cs_only": {"ics_weights":[1.0,0.0,0.0,0.0,0.0], "use_rep": False}
}

datasets = base.CONFIG.get("datasets", ["MNIST", "EMNIST", "CIFAR10"])

summary_rows = []

# Run grid
for ds in datasets:
    for vname, vcfg in variants.items():
        print(f"Running variant {vname} on dataset {ds} ...")
        orig_cfg = copy.deepcopy(base.CONFIG)   # deep copy so nested structures are preserved
        try:
            # set variant config
            base.CONFIG["ics_weights"] = vcfg["ics_weights"]
            base.CONFIG["use_rep"] = vcfg["use_rep"]

            # prepare output subdir
            run_out = os.path.join(ABL_OUT, f"{ds}__{vname}")
            if os.path.exists(run_out):
                shutil.rmtree(run_out)
            os.makedirs(run_out, exist_ok=True)

            # set save_dir to variant dir temporarily
            base.CONFIG["save_dir"] = run_out

            # run experiment (this will write logs + plots into run_out)
            start = time.time()
            df, ics_hist = base.run_experiment_for_dataset(ds)
            elapsed = time.time() - start

            final_acc = float(df["global_acc"].iloc[-1]) if "global_acc" in df.columns else None
            final_asr = float(df["ASR"].iloc[-1]) if "ASR" in df.columns else None
            summary_rows.append({"dataset":ds, "variant":vname, "final_acc":final_acc, "final_asr":final_asr, "time_s":elapsed, "out_dir":run_out})

            print(f"Finished {vname} on {ds} in {elapsed:.1f}s; results at {run_out}")

        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error running {vname} on {ds}: {e}\n{tb}")
            # still add a failed row
            summary_rows.append({"dataset":ds, "variant":vname, "final_acc":None, "final_asr":None, "time_s":None, "out_dir":run_out, "error": str(e)})

        finally:
            # restore orig config
            base.CONFIG = copy.deepcopy(orig_cfg)

# write summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(ABL_OUT, "ablation_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print("Saved ablation summary to", summary_csv)

# produce comparison plots across variants for each dataset
for ds in datasets:
    plt.figure(figsize=(6,4))
    plotted = False
    for vname in variants.keys():
        run_out = os.path.join(ABL_OUT, f"{ds}__{vname}")
        log_path = os.path.join(run_out, f"{ds}_logs.csv")
        if not os.path.exists(log_path):
            continue
        df = pd.read_csv(log_path)
        if 'round' in df.columns and 'global_acc' in df.columns:
            plt.plot(df['round'], df['global_acc'], label=vname)
            plotted = True
    if not plotted:
        plt.close(); continue
    plt.title(f"Ablation comparison — global accuracy ({ds})")
    plt.xlabel("Round"); plt.ylabel("Global Accuracy")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(ABL_OUT, f"{ds}_ablation_compare.png"), dpi=300)
    plt.close()

print("Ablation run complete. Outputs in:", ABL_OUT)
