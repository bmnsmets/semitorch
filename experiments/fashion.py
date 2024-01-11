import time
import os
import sys
from pathlib import Path
import typer
from typing import Annotated, Optional, Iterable, List, Union
from types import SimpleNamespace
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
import torchvision
import wandb
import timm
import torchmetrics


#### Printing
import rich
from rich import print
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn

console = rich.console.Console()
rich.traceback.install(show_locals=True)

####    Model import
from fashion_models import (
    ModelName,
    list_models,
    create_model,
    create_config,
    resetmodel,
)


PROJECT_NAME = "semitorch-convnext-fashionmnist"
DATA_ROOT = Path("./data/")


####
####    Utilities
####


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


####
####    Data preparation
####


def download(urls: Iterable[str], dest: Path):
    """
    Download the URLs to the destination path.
    """
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TaskID,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )
    from concurrent.futures import ThreadPoolExecutor
    from urllib.request import urlopen
    from functools import partial
    from pathlib import Path

    progress = Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    )

    def _download(task_id: TaskID, url: str, path: Path):
        progress.console.log(f"Requesting {url}")
        response = urlopen(url)
        content_length = int(response.info()["Content-length"])
        progress.update(task_id, total=content_length)
        with open(path, "wb") as file:
            progress.start_task(task_id)
            for data in iter(partial(response.read, 32768), b""):
                file.write(data)
                progress.update(task_id, advance=len(data))
        progress.console.log(f"Downloaded to {path}")

    with progress:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for url in urls:
                filename = url.split("/")[-1]
                path = dest / filename
                task_id = progress.add_task("download", filename=filename, start=False)
                pool.submit(_download, task_id, url, path)


def preload_fashionmnist(path=DATA_ROOT):
    """
    If not already present will download the FashionMNIST dataset
    so that it can be loaded by `torchvision.datasets.FashionMNIST`.
    """
    from urllib.error import URLError
    import hashlib
    from pathlib import Path
    import gzip
    import shutil

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]
    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    path = path / "FashionMNIST" / "raw"
    path.mkdir(parents=True, exist_ok=True)

    def _calc_md5(fpath: Path, chunk_size: int = 1024 * 1024) -> str:
        md5 = hashlib.md5(usedforsecurity=False)
        with open(fpath, "rb") as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()

    # check if already present
    if all([(path / (Path(fn).stem)).exists() for fn, _ in resources]):
        console.log(f"FashionMNIST already downloaded.")
        return

    # download and uncompress
    for filename, md5 in resources:
        (path / filename).unlink(missing_ok=True)
        for mirror in mirrors:
            url = f"{mirror}{filename}"
            try:
                download([url], path)
                file_md5 = _calc_md5(path / filename)
                if md5 != file_md5:
                    raise RuntimeError(
                        f"Download from {url} failed MD5 checksum, was expecting {md5} but got {file_md5}."
                    )
                with console.status(
                    f"Decompressing {filename}", spinner="dots", spinner_style="green"
                ):
                    time.sleep(2)
                    with gzip.open((path / filename), "rb") as f_in:
                        with open((path / (Path(filename).stem)), "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                console.log(f"Decompressed {filename}")

            except URLError as error:
                print(f"Failed to download (trying next mirror if available):\n{error}")
                continue
            break
        else:
            raise RuntimeError(f"Error downloading {filename}")

    return


def get_dataloaders(
    batch_size: int, num_workers: Optional[int] = None, rng_seed: Optional[int] = None
):
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    # Load FashionMNIST dataset
    transforms_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,)),
            # transforms.Resize((16, 16), antialias=True),
            # transforms.RandomResizedCrop(
            #     (16, 16), scale=(0.9, 1.0), ratio=(0.9, 1.1), antialias=True
            # ),
        ]
    )
    transforms_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,)),
            # transforms.Resize((16, 16), antialias=True),
        ]
    )
    fashion_train = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=True, transform=transforms_train
    )
    fashion_test = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=False, transform=transforms_test
    )

    # fashion_num_features = fashion_test[0][0].shape[1] * fashion_test[0][0].shape[1]
    # fashion_num_classes = torch.unique(fashion_test.targets).shape[0]

    if rng_seed:
        g = torch.Generator()
        g.manual_seed(rng_seed)
    else:
        g = None

    # create loaders
    fashion_train_loader = DataLoader(
        fashion_train,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        generator=g,
        num_workers=num_workers,
        sampler=DistributedSampler(fashion_train),
    )
    fashion_test_loader = DataLoader(
        fashion_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return fashion_train_loader, fashion_test_loader


####
####    Training
####


def train_batch(model: nn.Module):
    pass


def train(
    rank: int,
    cfg: SimpleNamespace,
    progress: dict,
    logqueue: mp.Queue,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "50100"
    init_process_group(backend="nccl", rank=rank, world_size=cfg.world_size)
    device = torch.device("cuda", cfg.devices[rank])
    torch.cuda.set_device(device)
    cfg.run_seed and torch.manual_seed(cfg.seed)

    def log(x):
        logqueue.put_nowait(
            (
                f"\[rank{rank}@"
                f"{sys._getframe().f_back.f_code.co_filename.split('/')[-1]}:"
                f"{sys._getframe().f_back.f_lineno}] {x}"
            )
        )

    log0 = lambda x: log(x) if rank == 0 else None

    train_loader, test_loader = get_dataloaders(
        cfg.batchsize, num_workers=4, rng_seed=cfg.run_seed
    )
    batches = len(train_loader)
    total_samples = len(train_loader.dataset)
    log0(f"Training dataset: {total_samples:,} samples in {batches:,} batches.")
    log0(f"Test dataset: {len(test_loader.dataset):,} samples.")

    model = DDP(create_model(cfg.modelname).to(device))
    log(f"Created DDP model '{cfg.modelname}' on {device}.")

    config = create_config(cfg.modelname, cfg.batchsize, cfg.epochs)

    if rank == 0:
        progress[rank] = {
            "epoch": 0,
            "total_epochs": cfg.epochs,
            "sample": 0,
            "total_samples": total_samples,
        }

    for i in range(cfg.epochs):
        for j in range(batches):
            time.sleep(0.0003)
            if rank == 0:
                progress[rank] = {
                    "epoch": i + 1,
                    "total_epochs": cfg.epochs,
                    "sample": (j + 1) * cfg.batchsize,
                    "total_samples": total_samples,
                }
    log0(cfg)
    destroy_process_group()
    return


####
####    Entrypoints
####


def main_local(cfg: SimpleNamespace):
    import queue

    # Check that CUDA is available.
    if not torch.cuda.is_available():
        raise RuntimeError(f"PyTorch reports CUDA is not available.")
    elif torch.cuda.device_count() == 0:
        raise RuntimeError(f"No CUDA devices available.")

    # Generate run seeds from the initial seeds for reproducibility if specified.
    if cfg.reproducible:
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(cfg.initial_seed)
        # generate manual seed for each run
        cfg.seeds = torch.randint(torch.iinfo(torch.long).max, (cfg.runs,)).tolist()
        console.log(f"Reproducible seeds: {cfg.seeds}.")
    else:  # no seed specified, don't care about reproducibility
        cfg.seeds = [None for i in range(cfg.runs)]
        console.log(f"Running in non-reproducible mode.")

    # Enumerate all devices if devices have not been manually specified.
    if not cfg.devices:
        cfg.devices = list(range(torch.cuda.device_count()))

    # Let user know what GPUs we are using.
    console.log(f"Using CUDA {torch.version.cuda} on the following devices:")
    for device in cfg.devices:
        props = torch.cuda.get_device_properties(device)
        console.log(
            (
                f"  {device}: {props.name} "
                f"(mem={props.total_memory / 1024**3:.1f} GB, "
                f"cc={props.major}.{props.minor})"
            )
        )

    # Number of worker processes to spawn.
    cfg.world_size = len(cfg.devices)
    if cfg.world_size == 1:
        console.log(f"Spawning {cfg.world_size} training process.")
    else:
        console.log(f"Spawning {cfg.world_size} training processes.")

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.completed}/{task.total}",
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ) as progress:
        sample_progress = progress.add_task("[purple]Samples:", visible=False)
        epoch_progress = progress.add_task("[blue]Epochs:", visible=False)
        overall_progress = progress.add_task("[green]Runs:", total=cfg.runs)
        for r in range(1, cfg.runs + 1):
            cfg.run = r
            cfg.run_seed = cfg.seeds[r - 1]
            with mp.Manager() as manager:
                _progress = manager.dict()
                _logqueue = manager.Queue(128)
                ctx = mp.spawn(
                    train,
                    args=(cfg, _progress, _logqueue),
                    nprocs=cfg.world_size,
                    join=False,
                )

                while any([p.is_alive() for p in ctx.processes]):
                    try:
                        str = _logqueue.get_nowait()
                        if str:
                            progress.console.log(str)
                    except queue.Empty as e:
                        pass
                    try:
                        epoch = _progress[0]["epoch"]
                        total_epochs = _progress[0]["total_epochs"]
                        sample = _progress[0]["sample"]
                        total_samples = _progress[0]["total_samples"]
                        progress.update(
                            epoch_progress,
                            completed=epoch,
                            total=total_epochs,
                            visible=True,
                        )
                        progress.update(
                            sample_progress,
                            completed=sample,
                            total=total_samples,
                            visible=True,
                        )
                        time.sleep(0.01)
                    except KeyError as e:
                        time.sleep(0.1)

                ctx.join()

            progress.update(overall_progress, advance=1)
    return


def main_modal(cfg: SimpleNamespace):
    if not torch.cuda.is_available():
        raise RuntimeError(f"PyTorch reports CUDA is not available.")
    elif torch.cuda.device_count() == 0:
        raise RuntimeError(f"No CUDA devices available.")

    if not cfg.devices:
        cfg.devices = range(torch.cuda.device_count())

    console.log(f"Using CUDA {torch.version.cuda} on the following devices:")
    for device in cfg.devices:
        props = torch.cuda.get_device_properties(device)
        console.log(
            (
                f"  {device}: {props.name} "
                f"(mem={props.total_memory / 1024**3:.0f} GB, "
                f"cc={props.major}.{props.minor})"
            )
        )

    world_size = len(cfg.devices)
    return


def main(
    modelname: Annotated[
        str,
        typer.Argument(
            help="Name of the model to train or 'list' to get a list of available models."
        ),
    ],
    batchsize: Annotated[
        int, typer.Option("--batchsize", "-b", help="Batchsize.")
    ] = 512,
    epochs: Annotated[
        int, typer.Option("--epochs", "-e", help="Number of epochs to train.")
    ] = 50,
    runs: Annotated[
        int, typer.Option("--runs", "-r", help="Number of experiments to run")
    ] = 1,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", help="Set RNG seed for reproducbility."),
    ] = None,
    use_local: Annotated[
        bool,
        typer.Option(
            "--local", help="Run training locally instead of the modal cloud."
        ),
    ] = False,
    use_wandb: Annotated[
        bool, typer.Option("--wandb", help="Log data about training to wandb.ai")
    ] = True,
    devices: Annotated[
        Optional[List[int]],
        typer.Option(
            "--device",
            "-d",
            help="Manually specify CUDA devices to use. If not specified all devices will be used.",
        ),
    ] = None,
):
    print(f"🧪 [bold]ConvNeXt/FashionMNIST training[/bold]")

    if modelname.lower() == "list":
        console.print(f"Available models:")
        for m in ModelName.__members__.values():
            console.print(f" {m}")
        return
    else:
        modelname = ModelName(modelname)

    config = SimpleNamespace()
    config.modelname = modelname
    config.batchsize = batchsize
    config.epochs = epochs
    config.runs = runs
    config.initial_seed = seed
    config.reproducible = seed != None
    config.platform = "local" if use_local else "modal"
    config.use_wandb = use_wandb
    config.devices = devices

    if use_local:
        preload_fashionmnist()
        main_local(config)
    else:
        import modal

        image = (
            modal.Image.debian_slim()
            .pip_install(
                "torch~=2.1.2",
                "torchvision~=0.16.2",
                "wandb~=0.16.1",
                "triton~=2.1.0",
                "typer~=0.9.0",
                "rich~=13.7.0",
                "timm~=0.9.12",
                "torchmetrics~=1.2.1",
            )
            .run_function(preload_fashionmnist)
        )

        secrets = []

        if use_wandb:
            secrets.append(modal.Secret.from_name("wandb"))

        stub = modal.Stub(
            "semitorch_convnext_fashionmnist", image=image, secrets=secrets
        )

        stub.function(gpu="t4")(main_modal)
        with stub.run():
            stub.main_modal.remote(config)

    return


if __name__ == "__main__":
    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        rich_markup_mode="rich",
    )
    app.command()(main)
    app()
