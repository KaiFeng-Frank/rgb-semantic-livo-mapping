"""Compare the adapter against executable functions extracted from pinned authors' source."""
import argparse
import ast
import gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .glfs_baseline import GLFS, official_mdece


def run(upstream, loss_path):
    torch.set_num_threads(1)
    torch.manual_seed(481)
    # Never execute the training script's top-level services, paths or credentials.
    tree = ast.parse((upstream/'calibration_experiments/train_combined_model.py').read_text())
    weight_arrays = {}
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == 'nll_weight_type':
            kind = ast.literal_eval(node.test.comparators[0])
            literals = [n for n in ast.walk(node) if isinstance(n, ast.List) and len(n.elts) == 21]
            weight_arrays[kind] = np.asarray(ast.literal_eval(literals[0]))
    np.testing.assert_allclose(weight_arrays['log'], 2*np.log(weight_arrays['sqrt']), atol=2e-8, rtol=0)
    np.testing.assert_allclose(np.exp(-weight_arrays['log']).sum(), 1., atol=2e-8, rtol=0)
    weight = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StreamlinedWeightEstimator')
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CombinedTrainer')
    forward = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    class CPU(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant('cpu'), node) if node.value == 'cuda:0' else node
    code = ast.fix_missing_locations(CPU().visit(ast.Module(body=[weight, forward], type_ignores=[])))
    ns = dict(torch=torch, nn=nn, np=np, gc=gc, sm=nn.Softmax(2), sm2=nn.Softmax(1), torch_device='cpu')
    exec(compile(code, '<author-functions-device-only-adaptation>', 'exec'), ns)
    scale_tree = ast.parse((upstream/'calibration_experiments/learned_calibrators.py').read_text())
    exec(compile(scale_tree, '<author-vector-calibrator>', 'exec'), ns)
    author_weight = ns['StreamlinedWeightEstimator']()
    author_scale = ns['DifferentiableVectorCalibrator'](device='cpu', T0=[.4]*21)
    author = SimpleNamespace(scaling=author_scale, weight_estimator=author_weight, L=0, epsilon=0)
    ours = GLFS(21, (0, 1, 2, 3, 4, 5))
    with torch.no_grad():
        author_weight.weights.uniform_(.05, 2)
        author_scale.T.uniform_(-1, 1)
        ours.weights.copy_(author_weight.weights)
        ours.log_temperature.copy_(author_scale.T)
    logits = torch.randn(61, 5, 21)
    valid = torch.rand(61, 5) > .25
    valid[:, 0] = True
    logits[~valid] = 0
    depth, angle = torch.rand(61, 5)*8, torch.rand(61, 5)*110-10
    depth[0] = torch.tensor([0, 1, 2, 5, 6.])
    angle[0] = torch.tensor([0, 30, 60, 90, 100.])
    expected, log_expected = ns['forward'](author, dict(logits=logits, depth=depth, angle=angle))
    actual = ours(logits, depth, angle, valid)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    y = torch.randint(21, (61,))
    nn.functional.nll_loss(log_expected, y).backward()
    nn.functional.nll_loss(actual.log(), y).backward()
    torch.testing.assert_close(ours.log_temperature.grad, author_scale.T.grad, atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(ours.weights.grad, author_weight.weights.grad, atol=2e-7, rtol=2e-5)
    # Class-zero is a real nuScenes class; uniform logits are a valid observation.
    uniform = GLFS()(torch.zeros(2, 1, 16), torch.zeros(2, 1), torch.zeros(2, 1), torch.ones(2, 1, dtype=torch.bool))
    torch.testing.assert_close(uniform, torch.full((2, 16), 1/16))
    loss_tree = CPU().visit(ast.parse(loss_path.read_text()))
    loss_ns = {}
    exec(compile(ast.fix_missing_locations(loss_tree), '<author-loss-device-only-adaptation>', 'exec'), loss_ns)
    original_loss = loss_ns['mDECE']('cpu', 10, t_a=100, t_b=.01, ignore_class=-1)
    adapted_loss = official_mdece(loss_path)
    p = torch.randn(80, 16).softmax(-1).requires_grad_()
    targets = torch.arange(80) % 16
    a, b = original_loss(p, targets), adapted_loss(p, targets)
    torch.testing.assert_close(a.squeeze(), b.squeeze(), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(torch.autograd.grad(a, p)[0], torch.autograd.grad(b, p)[0])
    return dict(passed=True, forward_max_abs=float((actual-expected).abs().max().detach()),
                checks=['author forward', 'temperature gradient', 'lookup-weight gradient',
                        'bin boundaries', 'explicit zero-logit mask', 'author mDECE value and gradient',
                        'released log-frequency class weights'])


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--loss', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.upstream, args.loss), indent=2))
