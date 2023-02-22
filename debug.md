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

Epoch 2: 100%|█| 8/8 [00:27<00:00,  3.45s/it, loss=0.649, v_num=0, train_accuracy_step=0.812, train_multiclass_auroc_step=0.290, train_binary_pf1_step=0.00533

3. Upsample, epoch = 1

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