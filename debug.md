# Debug

1. No upsample, epoch = 1
python main.py --config-name config \
general.dataset_stage=train \
general.debug=True \
general.device=cpu \
general.experiment_id=rsna_debug \
trainer.max_epochs=1 \
datamodule.upsample=0 \
datamodule.datamodule_class._target_=examples.image_classification.rsna_breast_cancer_detection.datamodule.RSNAUpsampleDataModule \
model.model_class._target_=src.models.model.TimmModelWithGeM

Epoch 0: 100%|█| 8/8 [00:44<00:00,  5.58s/it, loss=0.653, v_num=0, train_accuracy_step=0.688, train_multiclass_auroc_step=0.483

2. No upsample, epoch = 3
python main.py --config-name config \
general.dataset_stage=train \
general.debug=True \
general.device=cpu \
general.experiment_id=rsna_debug \
trainer.max_epochs=3 \
datamodule.upsample=0 \
datamodule.datamodule_class._target_=examples.image_classification.rsna_breast_cancer_detection.datamodule.RSNAUpsampleDataModule \
model.model_class._target_=src.models.model.TimmModelWithGeM

Epoch 2: 100%|█| 8/8 [00:27<00:00,  3.45s/it, loss=0.649, v_num=0, train_accuracy_step=0.812, train_multiclass_auroc_step=0.290 train_binary_pf1_step=0.00533

--- Inference ---

python main.py --config-name config \
general.dataset_stage=test \
general.device=cpu \
general.experiment_id=rsna_debug \
trainer.max_epochs=3 \
trainer.precision=32 \
datamodule.upsample=0 \
datamodule.datamodule_class._target_=examples.image_classification.rsna_breast_cancer_detection.datamodule.RSNAUpsampleDataModule \
model.model_class._target_=src.models.model.TimmModelWithGeM

float32
[[0.7353294  0.2646706 ]
 [0.6602107  0.33978924]
 [0.73217267 0.2678273 ]
 [0.3339309  0.66606915]]

 float16 bf16
[[0.72265625 0.27734375]
 [0.65234375 0.34765625]
 [0.73828125 0.26171875]
 [0.31445312 0.68359375]]

# pytorch
 0.0061])), ('backbone.layer4.1.bn2.num_batches_tracked', tensor(12))
('head.bias', tensor([0.0433, 0.0008]))])
# pytorch lightning
 ('model.head.bias', tensor([0.0427, 0.0014]))])

1. Upsample, epoch = 1

python main.py --config-name config \
general.dataset_stage=train \
general.debug=True \
general.device=cpu \
general.experiment_id=rsna_debug \
trainer.max_epochs=1 \
datamodule.upsample=3 \
datamodule.datamodule_class._target_=examples.image_classification.rsna_breast_cancer_detection.datamodule.RSNAUpsampleDataModule \
model.model_class._target_=src.models.model.TimmModelWithGeM

Epoch 0: 100%|█| 8/8 [00:26<00:00,  3.27s/it, loss=0.634, v_num=0, train_accuracy_step=0.688, train_multiclass_auroc_step=0.645

4. No upsample with balanced sampler

python main.py --config-name config \
general.dataset_stage=train \
general.debug=True \
general.device=cpu \
general.experiment_id=rsna_debug \
trainer.max_epochs=1 \
datamodule.upsample=0 \
datamodule.datamodule_class._target_=examples.image_classification.rsna_breast_cancer_detection.datamodule.RSNAUpsampleBalancedSamplerDataModule \
model.model_class._target_=src.models.model.TimmModelWithGeM

Epoch 0: 100%|█| 8/8 [00:25<00:00,  3.13s/it, loss=0.727, v_num=0, train_accuracy_step=0.688, train_binary_pf1_step=0.219

5. heng augs

python main.py --config-name rsna_heng \
general.dataset_stage=train \
general.debug=True \
general.device=cpu \
general.experiment_id=rsna_debug \
trainer.max_epochs=1 \
datamodule.upsample=0 \
datamodule.datamodule_class._target_=examples.image_classification.rsna_breast_cancer_detection.datamodule.RSNAUpsampleDataModule \
model.model_class._target_=src.models.model.TimmModelWithGeM