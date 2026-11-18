# LROC FM (MAE layout, Prithvi-EO-2.0 model structure)

```
main_pretrain.py, engine_pretrain.py   training (MAE)
models_mae.py                          model factories used by main_pretrain.py
models/lroc_mae.py                     ViT encoder, decoder, MAE (2D sin/cos, per-patch metadata)
models/lroc_vit.py                     configs, checkpoint loading, TerraTorch registration
util/                                  pos_embed, lr_sched, misc (MAE); config, geo, datasets (LROC)
tools/make_manifest.py                 process_edrs.sh outputs -> manifest.csv (+ alignment check)
tools/build_index.py                   chips, spatial 95/5 split, normalisation stats
configs/data.yaml                      data and metadata settings
tests/                                 python tests/<file>.py from the repo root
```

## Run (from the repo root, with the environment active)

```
bash tests/get_refs.sh
python tests/test_pos_embed.py && python tests/test_models_mae.py && python tests/test_lroc_vit.py

python tools/make_manifest.py --processed ~/ProcessedData --check 10
python tools/build_index.py --config configs/data.yaml --workers 16

CUDA_VISIBLE_DEVICES=0 python main_pretrain.py --data_config configs/data.yaml \
    --model lroc_mae_vit_tiny_patch16 --batch_size 64 --epochs 2 --warmup_epochs 1 \
    --output_dir runs/smoke --log_dir runs/smoke

torchrun --nproc_per_node=8 main_pretrain.py --data_config configs/data.yaml \
    --model lroc_mae_vit_base_patch16 --batch_size 128 --blr 1.5e-4 --epochs 400 --warmup_epochs 40 \
    --num_workers 8 --output_dir runs/vitb --log_dir runs/vitb
```
