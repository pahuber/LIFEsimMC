"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """LIFEsim 2."""


if __name__ == "__main__":
    main(prog_name="lifesimmc")  # pragma: no cover
