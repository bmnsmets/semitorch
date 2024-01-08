import time
from pathlib import Path
import typer
from typing import Annotated, Optional, Iterable
import torchvision
import wandb
from enum import Enum

DATA_ROOT = Path("./data/")

class ModelName(str, Enum):
    convnext_atto = "convnext_atto"
    convnext_maxplus_atto = "convnext_maxplus_atto"
    convnext_minplus_atto = "convnext_minplus_atto"
    convnext_log_m1_atto = "convnext_log_m1_atto"
    convnext_log_p1_atto = "convnext_log_p1_atto"


#### Printing
import rich

console = rich.console.Console()
rich.traceback.install(show_locals=True)


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


def foo():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 1))  # connect() for UDP doesn't send packets
    local_ip_address = s.getsockname()[0]
    console.log(f"Hello from {socket.gethostname()} ({local_ip_address})")


####
####    Training
####


def train_batch():
    pass


####
####    Entrypoint
####


def main(
    model: Annotated[ModelName, typer.Argument(help="Name of the model to train.")],
    epochs: Annotated[
        int, typer.Option("--epochs", "-e", help="Number of epochs to train.")
    ] = 50,
    runs: Annotated[
        int, typer.Option("--runs", "-r", help="Number of experiments to run")
    ] = 10,
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Set RNG seed for reproducbility.")
    ] = 42,
    use_local: Annotated[
        bool,
        typer.Option(
            "--local", help="Run training locally instead of the modal cloud."
        ),
    ] = False,
    use_wandb: Annotated[
        bool, typer.Option("--wandb", help="Log data about training to wandb.ai")
    ] = True,
):
    print(f"🧪 [bold]ConvNeXt/FashionMNIST training[/bold]")

    if use_local:
        preload_fashionmnist()
        foo()
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
            )
            .run_function(preload_fashionmnist)
        )

        if use_wandb:
            stub = modal.Stub(
                "semitorch_convnext_fashionmnist",
                image=image,
                secret=modal.Secret.from_name("wandb"),
            )
        else:
            stub = modal.Stub("semitorch_convnext_fashionmnist", image=image)

        stub.function(gpu="t4")(foo)
        with stub.run():
            stub.foo.remote()

    return


if __name__ == "__main__":
    typer.run(main)