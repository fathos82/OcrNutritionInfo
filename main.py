import argparse

from structure.application import Application
from structure.pipeline_factory import PipelineVariation
from structure.runners.runner_configuration import RunnerConfiguration


def parse_arguments():
    """Configura e retorna os argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description="Configuração do Runner de Pipeline")

    parser.add_argument(
        "--processing-name",
        required=True,
        help="Nome do processamento"
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=[str(p.value ) for p in PipelineVariation],
        help="Tipo de pipeline a ser executado"
    )
    parser.add_argument(
        "--runner",
        required=True,
        choices=['server', 'local', 'client'],
        help="Ambiente de execução"
    )

    return parser.parse_args()


def create_configuration(args):
    """Cria uma instância de RunnerConfiguration a partir dos argumentos"""
    return RunnerConfiguration(
        processing_name=args.processing_name,
        pipeline=args.pipeline,
        runner=args.runner
    )


def main():
    # Parse arguments
    args = parse_arguments()

    # Create configuration
    config = create_configuration(args)
    Application(config).run()



if __name__ == "__main__":
    main()