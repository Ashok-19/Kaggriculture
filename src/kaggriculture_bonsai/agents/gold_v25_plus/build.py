"""Build the isolated CPU/CUDA reinforcement evaluator into disposable /tmp."""
import subprocess
from kaggriculture_bonsai.agents.gold_v25.extract import ROOT


def main():
    common = ['-std=c++17', '-O2', '-Inative/bkg4/include', '-Inative/bkg4/stateful']
    source = 'src/kaggriculture_bonsai/agents/gold_v25_plus/evaluate.cu'
    for command in (
        ['g++', *common, '-x', 'c++', '-Inative/cpp/include', source, '-o', '/tmp/gold_plus_cpu'],
        ['nvcc', *common, '-arch=sm_86', '-Inative/cuda/include', source, '-o', '/tmp/gold_plus_cuda'],
    ):
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
