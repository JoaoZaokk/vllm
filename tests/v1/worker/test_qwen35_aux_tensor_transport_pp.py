# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aux taps must cross the PP boundary with the same VALUES, in the same ORDER.

The sibling file `test_eagle3_aux_hidden_states_pp.py` says what it is up front:
"Accounting checks". It verifies which tap *ids* each stage claims. It never
moves a tensor, so it cannot see a transport that carries the right count of the
wrong things -- taps swapped between slots, a stage's taps overwriting another's,
or a residual dropped on the way.

That gap has a name in this project. The greedy equivalence run of 19/ago proved
the speculative VERIFIER is correct under PP=2: it never accepts a token the
target would not produce. It does NOT prove the taps are right. A wrong tap
makes the drafter propose badly, the verifier correctly rejects, and the only
symptom is lower acceptance -- a number nobody can call wrong by looking at it.

So this file exercises the real transport with real tensors:

    pack_local_aux_for_last   -> what a non-last stage puts on the wire
    recv_remote_aux_from_producers -> what the last stage reads back
    remote + local            -> the concatenation in qwen3_next.py:735

and asserts the last stage ends up with exactly what a single stage would have
produced. No model is loaded; the tensors carry values that identify which tap
produced them, so order is checkable and not just length.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.models.interfaces import EagleModelMixin

# Qwen3.5/3.8-27B: 64 layers. The checkpoint declares target_layer_ids
# [4, 16, 28, 40, 52]; get_eagle3_aux_layers_from_config maps them +1.
NUM_LAYERS = 64
AUX_IDS = (5, 17, 29, 41, 53)
DIM = 8


class _Estagio(EagleModelMixin):
    """Smallest object the transport needs: a layer count and a tap list."""

    def __init__(self, rank: int, pp: int, num_layers=NUM_LAYERS, aux_ids=AUX_IDS):
        self.config = SimpleNamespace(num_hidden_layers=num_layers)
        self.aux_hidden_state_layers = aux_ids
        self.rank, self.pp = rank, pp
        # _cache_aux_pp_layout() reads the live PP group, which does not exist
        # outside a worker. The values it would compute are these.
        self._aux_slot_base_cached = self._aux_slot_base(rank, pp) if pp > 1 else 0
        self._aux_upstream_total_cached = self._aux_slot_base(pp - 1, pp) if pp > 1 else 0

    def taps_locais(self):
        inicio, fim = get_pp_indices(self.config.num_hidden_layers, self.rank, self.pp)
        return self.local_aux_tap_ids(
            inicio, fim, tuple(self.aux_hidden_state_layers), self.rank == 0)


class _BufferFalso:
    """Stand-in for IntermediateTensors: a dict with __getitem__."""

    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, chave):
        return self.tensors[chave]


def _tensor_do_tap(tap_id: int) -> torch.Tensor:
    """Value encodes the producing tap, so a swap is visible, not just a count."""
    return torch.full((2, DIM), float(tap_id))


def _colher(pp: int, num_layers=NUM_LAYERS, aux_ids=AUX_IDS) -> list[torch.Tensor]:
    """Run the real transport across `pp` stages, return what the last one sees."""
    buffer: dict[str, torch.Tensor] = {}
    for rank in range(pp):
        estagio = _Estagio(rank, pp, num_layers, aux_ids)
        locais = [_tensor_do_tap(t) for t in estagio.taps_locais()]
        if rank < pp - 1:
            pacote = estagio.pack_local_aux_for_last(locais)
            # A collision here would silently drop a tap; assert instead.
            colidiu = set(pacote) & set(buffer)
            assert not colidiu, f"rank {rank} reused slots {sorted(colidiu)}"
            buffer.update(pacote)
        else:
            remoto = estagio.recv_remote_aux_from_producers(_BufferFalso(buffer))
            # Mirrors qwen3_next.py:735 -- upstream first, then this stage's own.
            return remoto + locais
    raise AssertionError("pp must be >= 1")


# --------------------------------------------------------------------------
# The property that matters
# --------------------------------------------------------------------------
@pytest.mark.parametrize("pp", [2, 3, 4, 8])
def test_split_preserva_valores_e_ordem_dos_taps(pp):
    """Splitting the model must not change what the drafter is fed."""
    referencia = _colher(1)
    dividido = _colher(pp)
    assert len(dividido) == len(referencia), (
        f"pp={pp} entregou {len(dividido)} taps contra {len(referencia)}")
    for i, (esperado, obtido) in enumerate(zip(referencia, dividido)):
        torch.testing.assert_close(
            obtido, esperado,
            msg=lambda m, i=i: (
                f"tap na posicao {i} mudou ao dividir em pp={pp}. "
                f"Esperado o tap {referencia[i][0, 0].item():.0f}, "
                f"veio {dividido[i][0, 0].item():.0f}. {m}"))


def test_a_referencia_tem_os_taps_do_checkpoint():
    """Guard on the guard: if pp=1 stopped producing the five taps, the test
    above would compare two wrong things and pass."""
    ids = [int(t[0, 0].item()) for t in _colher(1)]
    assert ids == list(AUX_IDS), ids


@pytest.mark.parametrize("pp", [2, 3, 4, 8])
def test_cada_tap_aparece_uma_vez_so(pp):
    """A duplicated tap keeps the count right when another is missing."""
    ids = [int(t[0, 0].item()) for t in _colher(pp)]
    assert sorted(ids) == sorted(set(ids)), f"tap duplicado em pp={pp}: {ids}"
    assert ids == sorted(ids), f"taps fora de ordem crescente em pp={pp}: {ids}"


@pytest.mark.parametrize("pp", [2, 4])
def test_slots_dos_estagios_nao_se_sobrepoem(pp):
    """Two stages writing the same slot loses a tap without any error: the
    second dict update simply wins."""
    vistos: dict[str, int] = {}
    for rank in range(pp - 1):
        estagio = _Estagio(rank, pp)
        for chave in estagio.pack_local_aux_for_last(
                [_tensor_do_tap(t) for t in estagio.taps_locais()]):
            assert chave not in vistos, (
                f"slot {chave} escrito por rank {vistos[chave]} e por {rank}")
            vistos[chave] = rank


# --------------------------------------------------------------------------
# The failure the code already guards, kept honest
# --------------------------------------------------------------------------
def test_slot_ausente_levanta_em_vez_de_zerar():
    """The production comment says it: substituting zeros here would cost
    acceptance without failing. That is the worst kind of bug in this project --
    a number that is wrong and does not announce itself."""
    estagio = _Estagio(1, 2)
    with pytest.raises(RuntimeError, match="missing from the last stage"):
        estagio.recv_remote_aux_from_producers(_BufferFalso({}))


def test_pp1_nao_espera_nada_de_upstream():
    assert _Estagio(0, 1).recv_remote_aux_from_producers(None) == []


# --------------------------------------------------------------------------
# The tap value itself
# --------------------------------------------------------------------------
def test_tap_soma_o_residual():
    """A tap that forgets the residual is a silently wrong tensor: same shape,
    same dtype, same position, different content."""
    estagio = _Estagio(0, 1)
    h = torch.ones(2, DIM)
    r = torch.full((2, DIM), 3.0)
    saida = estagio._maybe_add_hidden_state([], AUX_IDS[0], h, r)
    assert len(saida) == 1
    torch.testing.assert_close(saida[0], h + r)


def test_camada_que_nao_e_tap_nao_entra():
    estagio = _Estagio(0, 1)
    nao_tap = next(i for i in range(NUM_LAYERS) if i not in AUX_IDS)
    assert estagio._maybe_add_hidden_state([], nao_tap, torch.ones(2, DIM), None) == []


# --------------------------------------------------------------------------
# Partition changes what is being tested -- so it is checked, not assumed
# --------------------------------------------------------------------------
@pytest.mark.parametrize("particao,esperado", [
    ("16,48", (1, 4)),
    ("20,44", (2, 3)),
    ("32,32", (3, 2)),
])
def test_particao_decide_quantos_taps_cruzam_a_fronteira(monkeypatch, particao, esperado):
    """Reparticionar nao e ajuste de memoria: muda o que a equivalencia testa.

    O perfil dspark_pp2 afirma que com 16,48 exatamente UM tap fica no estagio 0
    e quatro atravessam. O veredito IDENTICO de 19/ago foi medido nessa
    condicao. Se get_pp_indices ou a lista de taps mudarem, aquela nota vira
    mentira e o veredito passa a descrever outra topologia.

    Precisa do env: sem VLLM_PP_LAYER_PARTITION o split e' uniforme (32/32), e
    nao e' o que o laboratorio roda.
    """
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", particao)
    from vllm import envs
    monkeypatch.setattr(envs, "VLLM_PP_LAYER_PARTITION", particao, raising=False)

    primeiros = EagleModelMixin.local_aux_tap_ids(
        *get_pp_indices(NUM_LAYERS, 0, 2), AUX_IDS, True)
    ultimos = EagleModelMixin.local_aux_tap_ids(
        *get_pp_indices(NUM_LAYERS, 1, 2), AUX_IDS, False)
    assert (len(primeiros), len(ultimos)) == esperado, (primeiros, ultimos)
    # e o conjunto continua completo: reparticionar move taps, nao os perde
    assert sorted(primeiros + ultimos) == list(AUX_IDS)
