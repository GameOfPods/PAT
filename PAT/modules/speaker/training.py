#  PAT - Toolkit to analyze podcast audio and topics talked about in the podcast. For example Books
#  Copyright (c) 2024.  RedRem95
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

import os


def main(
        target_folder: str,
        database_path: str,
        database_name: str,
        protocol: str,
        diarization_pipe: str,
        segmentation_model: str,
        epochs: int,
        batch_size: int,
        optim_iters: int = None,
        model_duration: int = None,
        num_workers: int = None,
):
    if optim_iters is None:
        optim_iters = epochs
    if database_path is not None:
        os.environ["PYANNOTE_DATABASE_CONFIG"] = database_path
        print(f"Added database-path \"{database_path}\" to environment as \"PYANNOTE_DATABASE_CONFIG\".")

    print(f"Will finetune \"{diarization_pipe}\" by training the model \"{segmentation_model}\" for {epochs} epochs "
          f"and then optimize for {optim_iters} iterations. Final output will be saved to \"{target_folder}\"")

    from types import MethodType
    from pprint import pformat
    from time import perf_counter
    import json
    import shutil
    import pathlib

    from tqdm import tqdm
    import pytorch_lightning as pl
    from lightning.pytorch import loggers as pl_loggers
    from pyannote.database import registry, FileFinder
    from pyannote.database.protocol import SpeakerDiarizationProtocol as SDProtocol
    from pyannote.audio import Model, Pipeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    from pyannote.audio.tasks import Segmentation
    from torch.optim import Adam
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        RichProgressBar,
        DeviceStatsMonitor,
    )
    from pyannote.audio.pipelines import SpeakerDiarization
    from pyannote.pipeline import Optimizer
    from pyannote.database.protocol.protocol import ProtocolFile

    def fmt_secs(s: float):
        from datetime import timedelta
        return str(timedelta(seconds=s))

    available_tasks = registry.get_database(database_name).get_tasks()
    target_task = "SpeakerDiarization"
    print(f"Available tasks: {available_tasks} [{target_task} {'' if target_task in available_tasks else 'not '}found]")
    if target_task not in available_tasks:
        exit()

    protocol_data: SDProtocol = registry.get_protocol(
        f"{database_name}.{target_task}.{protocol}",
        preprocessors={"audio": FileFinder(registry=registry)}
    )

    print(f"loaded protocol {protocol_data.name}")

    try:
        train_size = len(list(protocol_data.train()))
        print(f"Train size: {train_size} - {sum(len(x['annotation']) for x in protocol_data.train())}")
        test_size = len(list(protocol_data.test()))
        print(f"Test  size: {test_size} - {sum(len(x['annotation']) for x in protocol_data.test())}")
        dev_size = len(list(protocol_data.development()))
        print(f"Dev   size: {dev_size} - {sum(len(x['annotation']) for x in protocol_data.development())}")
    except NotImplementedError as e:
        print(e)
        exit(1)

    print(f"Loading pretrained pipeline {diarization_pipe}")
    auth_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN", None)
    pretrained_pipeline = Pipeline.from_pretrained(diarization_pipe, use_auth_token=auth_token)
    metric_p = DiarizationErrorRate()
    for i, file in tqdm(enumerate(protocol_data.test()), total=test_size, desc="Testing", leave=True, unit="f"):
        file: ProtocolFile
        file["pretrained pipeline"] = pretrained_pipeline(file)
        metric_p(file["annotation"], file["pretrained pipeline"], uem=file["annotated"])
        if i >= test_size:
            break

    print(f"The pretrained pipeline reaches a Diarization Error Rate (DER) of {100 * abs(metric_p):.1f}% on test set.")

    print(f"Finetuning {segmentation_model} for max {epochs} epochs")
    t1 = perf_counter()
    model = Model.from_pretrained(segmentation_model, use_auth_token=auth_token)
    task = Segmentation(
        protocol_data,
        duration=model.specifications.duration if model_duration is None else model_duration,
        max_num_speakers=len(model.specifications.classes),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        loss="bce",
        vad_loss="bce")
    model.task = task
    model.setup(stage="fit")
    model.configure_optimizers = MethodType(lambda s: Adam(s.parameters(), lr=1e-4), model)
    monitor, direction = task.val_monitor
    checkpoint = ModelCheckpoint(
        dirpath=target_folder,
        monitor=monitor,
        mode=direction,
        save_top_k=5,
        every_n_epochs=1,
        save_last=False,
        save_weights_only=False,
        filename="podcast-model-{epoch:03d}-{val_loss:.2f}",
        verbose=False,
    )
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=direction,
        min_delta=0.0,
        patience=10,
        strict=True,
        verbose=False,
    )
    callbacks = [RichProgressBar(), DeviceStatsMonitor(cpu_stats=True), checkpoint, early_stopping]
    tb_logger: pl_loggers.Logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(target_folder, "tb_logs"))
    trainer = pl.Trainer(
        accelerator="auto", callbacks=callbacks, max_epochs=epochs, gradient_clip_val=0.5, devices=1, logger=[tb_logger]
    )
    t2 = perf_counter()
    trainer.fit(model)
    t3 = perf_counter()
    print(f"Training took {fmt_secs(t3 - t1)} (Setup: {fmt_secs(t2 - t1)}; Fitting: {fmt_secs(t3 - t2)})")

    best_model = os.path.join(target_folder, f"best{pathlib.Path(checkpoint.best_model_path).suffix}")
    shutil.copy(checkpoint.best_model_path, best_model)
    print(f"Best model: {best_model} from {checkpoint.best_model_path}")

    pretrained_hyperparameters = pretrained_pipeline.parameters(instantiated=True)
    print(f"Trained hyperparameters:\n{pformat(pretrained_hyperparameters)}")

    dev_set = list(protocol_data.development())
    optim_fmt_line = (
            "{i:%d}/%d: Best {task} threshold so far: {thresh}; Loss {loss}" % (len(str(optim_iters)), optim_iters)
    )

    print(f"Optimizing segmentation for {optim_iters} iterations")
    t1 = perf_counter()
    pipeline = SpeakerDiarization(
        segmentation=best_model,
        clustering="OracleClustering",
    )
    pipeline.freeze({"segmentation": {"min_duration_off": 0.0}})
    optimizer = Optimizer(pipeline)
    iterations = optimizer.tune_iter(
        dev_set,
        show_progress={"desc": "Optimizing segmentation", "leave": False, "unit": "i"}
    )
    best_loss = 1.0
    t2 = perf_counter()
    for i, iteration in enumerate(iterations, 1):
        print(optim_fmt_line.format(
            i=i, task="segmentation", thresh=iteration['params']['segmentation']['threshold'], loss=iteration["loss"], )
        )
        if i >= optim_iters:
            break
    t3 = perf_counter()
    best_segmentation_threshold = optimizer.best_params["segmentation"]["threshold"]
    print(f"Optimization took {fmt_secs(t3 - t1)} (Setup: {fmt_secs(t2 - t1)}; Tuning: {fmt_secs(t3 - t2)})")

    print(f"Optimizing clustering for {optim_iters} iterations")
    t1 = perf_counter()
    pipeline = SpeakerDiarization(
        segmentation=best_model,
        embedding=pretrained_pipeline.embedding,
        embedding_exclude_overlap=pretrained_pipeline.embedding_exclude_overlap,
        clustering=pretrained_pipeline.klustering,
    )
    pipeline.freeze({
        "segmentation": {
            "threshold": best_segmentation_threshold,
            "min_duration_off": 0.0,
        },
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 15,
        },
    })
    optimizer = Optimizer(pipeline)
    iterations = optimizer.tune_iter(
        dev_set,
        show_progress={"desc": "Optimizing clustering", "leave": False, "unit": "i"}
    )
    best_loss = 1.0
    t2 = perf_counter()
    for i, iteration in enumerate(iterations):
        print(optim_fmt_line.format(
            i=i, task="clustering", thresh=iteration['params']['clustering']['threshold'], loss=iteration["loss"], )
        )
        if i >= optim_iters:
            break
    t3 = perf_counter()
    best_clustering_threshold = optimizer.best_params['clustering']['threshold']
    print(f"Optimization took {fmt_secs(t3 - t1)} (Setup: {fmt_secs(t2 - t1)}; Tuning: {fmt_secs(t3 - t2)})")

    print("Building finetuned model")
    finetuned_pipeline = SpeakerDiarization(
        segmentation=best_model,
        embedding=pretrained_pipeline.embedding,
        embedding_exclude_overlap=pretrained_pipeline.embedding_exclude_overlap,
        clustering=pretrained_pipeline.klustering,
    )
    optimized_data = {
        "segmentation": {
            "threshold": best_segmentation_threshold,
            "min_duration_off": 0.0,
        },
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 15,
            "threshold": best_clustering_threshold,
        },
    }
    finetuned_pipeline = finetuned_pipeline.instantiate(optimized_data)

    optimized_data_path = os.path.join(target_folder, "optimized_data.json")
    with open(optimized_data_path, "w") as f_out:
        json.dump(optimized_data, f_out)
    print(f"Saved optimized data: {os.path.join(target_folder, 'optimized_data.json')}")

    metric_f = DiarizationErrorRate()

    for i, file in tqdm(enumerate(protocol_data.test()), total=test_size, desc="Testing", leave=True, unit="f"):
        file["finetuned pipeline"] = finetuned_pipeline(file)
        metric_f(file["annotation"], file["finetuned pipeline"], uem=file["annotated"])
        if i >= test_size:
            break

    print(f"The finetuned pipeline reaches a Diarization Error Rate (DER) of {100 * abs(metric_f):.1f}% on test set.")
    print("vs")
    print(f"The pretrained pipeline reaches a Diarization Error Rate (DER) of {100 * abs(metric_p):.1f}% on test set.")
    print(f"Important files: Best model is \"{best_model}\" from \"{checkpoint.best_model_path}\"")
    print(f"Optimized data is \"{optimized_data_path}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    from multiprocessing import cpu_count
    from math import ceil

    parser = ArgumentParser(description="Train speaker diarization model")
    parser.add_argument("-t", "--target", type=str, required=True, dest="target", help="target chkp file")
    parser.add_argument("-p", "--path", default=None, dest="path", help="Path to the database file")
    parser.add_argument("--database", default="PodcastDatabase", dest="database",
                        help="Database name [Default: %(default)s]")
    parser.add_argument("--protocol", default="Protocol", dest="protocol", help="Protocol name [Default: %(default)s]")
    parser.add_argument("--diarization_pipeline", default="pyannote/speaker-diarization-3.1", dest="diarization",
                        help="Diarization pipeline [Default: %(default)s]")
    parser.add_argument("--segmentation_model", default="pyannote/segmentation", dest="segmentation",
                        help="Segmentation model [Default: %(default)s]")
    parser.add_argument("--epochs", default=50, dest="epochs", type=int, help="Number of epochs [Default: %(default)s]")
    parser.add_argument("--optim-steps", default=None, type=int, dest="optim_steps",
                        help="optimization steps [Default: %(default)s]")
    parser.add_argument("--model-duration", default=None, type=int, dest="model_duration",
                        help="model duration [Default: %(default)s]")
    parser.add_argument("--num-workers", default=ceil(cpu_count() / 2), type=int, dest="num_workers",
                        help="Amount of training workers [Default: %(default)s]")
    parser.add_argument("--batch-size", default=32, type=int, dest="batch_size",
                        help="Batchsize [Default: %(default)s]")

    args = parser.parse_args()

    main(
        target_folder=args.target,
        database_path=args.path,
        database_name=args.database,
        protocol=args.protocol,
        diarization_pipe=args.diarization,
        segmentation_model=args.segmentation,
        epochs=args.epochs,
        optim_iters=args.optim_steps,
        model_duration=args.model_duration,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
