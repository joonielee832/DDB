_base_ = [
    "../../_base_/default_runtime.py",
    "../../_base_/models/deeplabv2red_r101v1c-d8_adapter.py",
    "../../_base_/datasets/50/ckd_gta2city_512x512.py",
    # Basic UDA Self-Training
    "../../_base_/uda/ckd.py",
    # AdamW Optimizer
    "../../_base_/schedules/adamw_ckd.py",
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    "../../_base_/schedules/poly10warm.py",
    # Schedule
    "../../_base_/schedules/schedule_40k.py",
]
uda = dict(
    cu_model_load_from="/home/results/50/checkpoints/r1_st_cu_r101.pth",
    ca_model_load_from="/home/results/50/checkpoints/r1_st_ca_r101.pth",
    cu_proto_path="/home/results/50/prototypes/gta2city_st-cu_dlv2.pth",
    ca_proto_path="/home/results/50/prototypes/gta2city_st-ca_dlv2.pth",
    proto_rectify=True,
)
# Random Seed
seed = 0
n_gpus = 2
runner = dict(
    type="DynamicIterBasedRunner", is_dynamic_ddp=True, pass_training_status=True
)

# Meta Information for Result Analysis
exp = "ckd"
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)
