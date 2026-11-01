"""CPU regressions for the RF DDP entry and its existing forward redirect."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from cosmos_predict2._src.imaginaire.utils.distributed import DistributedDataParallel, ddp_sync_grad
from cosmos_predict2._src.imaginaire.utils.ema import FastEmaModelUpdater
from cosmos_predict2._src.predict2.action.configs.action_conditioned.model import DDP_RECTIFIED_FLOW_CONFIG
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model import (
    ActionVideo2WorldModelRectifiedFlow,
)
from cosmos_predict2._src.predict2.models import text2world_model_rectified_flow as rf
from cosmos_predict2._src.predict2.utils import model_loader


def test_ddp_registration_preserves_rectified_flow():
    config = DDP_RECTIFIED_FLOW_CONFIG
    assert config['model']['_target_'] is ActionVideo2WorldModelRectifiedFlow
    assert config['model']['config'].fsdp_shard_size == 1
    assert config['trainer']['distributed_parallelism'] == 'ddp'
    assert config['trainer']['ddp'] == dict(static_graph=False, find_unused_parameters=True,
                                         broadcast_buffers=False, gradient_as_bucket_view=True)


def test_cpu_materialized_ema_starts_from_net():
    model = rf.Text2WorldModelRectifiedFlow.__new__(rf.Text2WorldModelRectifiedFlow)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(conditioner=None, fsdp_shard_size=1,
                                   ema=SimpleNamespace(enabled=True, rate=0.05))
    model.fsdp_device_mesh = None
    net, ema = nn.Linear(3, 2), nn.Linear(3, 2)
    model.build_net = mock.Mock(side_effect=[net, ema])
    with mock.patch.object(rf, 'lazy_instantiate', return_value=nn.Identity()):
        model.set_up_model()
    for source, target in zip(net.parameters(), ema.parameters()):
        torch.testing.assert_close(source, target, rtol=0, atol=0)
        assert not target.requires_grad


def test_consolidated_ddp_initializes_ema_after_weights_are_loaded():
    net, ema = nn.Linear(3, 2), nn.Linear(3, 2)
    net.register_buffer('position', torch.zeros(3, dtype=torch.bfloat16))
    ema.register_buffer('position', torch.empty(3, dtype=torch.float32))
    net.register_buffer('sample_counter', torch.tensor(0, dtype=torch.int64))
    ema.register_buffer('sample_counter', torch.empty((), dtype=torch.int64))
    model = SimpleNamespace(config=SimpleNamespace(ema=SimpleNamespace(enabled=True)),
                            net=net, net_ema=ema, net_ema_worker=FastEmaModelUpdater())
    model.cuda = lambda: model
    config = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(fsdp_shard_size=1)),
                             trainer=SimpleNamespace(distributed_parallelism='ddp'),
                             checkpoint=SimpleNamespace(load_path='net_only.pt', load_ema_to_reg=False))

    def load(**kwargs):
        for parameter in net.parameters():
            parameter.data.fill_(3)
        net.position.fill_(0.3)
        net.sample_counter.fill_(17)
        return model

    with mock.patch.object(model_loader, 'instantiate', return_value=model), \
            mock.patch.object(model_loader, 'load_model_state_dict_from_checkpoint', side_effect=load):
        assert model_loader.create_model_from_consolidated_checkpoint_with_fsdp(config) is model
    for source, target in zip(net.parameters(), ema.parameters()):
        torch.testing.assert_close(source, target, rtol=0, atol=0)
    for name, source in net.named_buffers():
        target = dict(ema.named_buffers())[name]
        torch.testing.assert_close(source.to(target.dtype), target, rtol=0, atol=0)


class TinyWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 3)

    def training_step(self, data):
        prediction = self.net(data[0])
        return {'prediction': prediction}, (prediction - data[1]).square().mean()


def accumulation_worker(rank, store, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{store}', rank=rank, world_size=2)
    try:
        torch.manual_seed(2026 + rank)
        model = TinyWorld()
        ddp = DistributedDataParallel(model, static_graph=False, find_unused_parameters=True,
                                      broadcast_buffers=False, gradient_as_bucket_view=True)
        reference = TinyWorld()
        reference.load_state_dict(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        generator = torch.Generator().manual_seed(777)
        for update in range(2):
            x, y = torch.randn(16, 4, generator=generator), torch.randn(16, 3, generator=generator)
            for micro in range(4):
                start = rank * 8 + micro * 2
                with ddp_sync_grad(ddp, micro == 3):
                    _, loss = ddp.training_step((x[start:start + 2], y[start:start + 2]))
                    (loss / 4).backward()
            _, reference_loss = reference.training_step((x, y))
            reference_loss.backward()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
            optimizer.step()
            reference_optimizer.step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
        torch.save(model.state_dict(), Path(output) / f'rank{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_two_rank_accumulation_matches_full_global16(tmp_path):
    mp.spawn(accumulation_worker, args=(str(tmp_path / 'store'), str(tmp_path)), nprocs=2, join=True)
    first = torch.load(tmp_path / 'rank0.pt', weights_only=True)
    second = torch.load(tmp_path / 'rank1.pt', weights_only=True)
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
