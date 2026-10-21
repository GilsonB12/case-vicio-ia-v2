"""Ponto de entrada protegido para rodar o CLI por script (necessário para o pool de processos no Windows)."""
import sys

from sistema import cli

if __name__ == "__main__":
    raise SystemExit(cli.main(sys.argv[1:]))
