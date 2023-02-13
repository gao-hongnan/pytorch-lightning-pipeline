def create_oof_df(pipeline_config: PipelineConfig) -> None:
    """Create OOF dataframe for Generic Image Dataset with a Resampling Strategy."""
    num_classes = pipeline_config.model.num_classes
    weights = [
        "/Users/reighns/gaohn/peekingduck-trainer/stores/model_artifacts/CIFAR-10/74502c5e-d25e-48c2-8b86-a690d33372f8/resnet18_best_val_Accuracy_fold_None_epoch9.pt"
    ]
    dm = ImageClassificationDataModule(pipeline_config)
    df_oof = pd.DataFrame()
    for fold in range(pipeline_config.resample.resample_params["n_splits"]):
        fold = fold + 1  # since fold starts from 1
        dm.prepare_data(fold=fold)
        _df_oof = dm.oof_df
        weight = weights[fold - 1]
        states = torch.load(weight)
        oof_probs = states["oof_probs"]
        oof_trues = states["oof_trues"]
        oof_preds = states["oof_preds"]

        _df_oof[[f"class_{str(c)}_oof" for c in range(num_classes)]] = (
            oof_probs.detach().cpu().numpy()
        )
        _df_oof["oof_trues"] = oof_trues.detach().cpu().numpy()
        _df_oof["oof_preds"] = oof_preds.detach().cpu().numpy()
        print(_df_oof.head())

        df_oof = pd.concat([df_oof, _df_oof], axis=0)

    oof_probs = torch.from_numpy(
        df_oof[[f"class_{str(c)}_oof" for c in range(num_classes)]].values
    )
    oof_trues = torch.from_numpy(df_oof["oof_trues"].values)

    accuracy = Accuracy(num_classes=num_classes)(oof_probs, oof_trues)
    print("OOF Accuracy", accuracy)  # 0.3281 confirms that it is the best epoch
