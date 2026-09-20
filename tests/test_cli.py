import pytest
from click.testing import CliRunner
from gene_language.cli.main import cli


def test_cli_help():
    runner = CliRunner()
    res = runner.invoke(cli, ["--help"])
    assert res.exit_code == 0
    assert "GENE-LANGUAGE-ANALYTICS" in res.output


def test_cli_analyze():
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("test.fasta", "w") as f:
            f.write(">test_seq\nATGCGATCGATCGATCGATC\n")
        res = runner.invoke(cli, ["analyze", "test.fasta", "--json-out"])
        assert res.exit_code == 0
        assert "shannon_entropy_bits" in res.output
