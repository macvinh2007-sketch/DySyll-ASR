"""DySyll-ASR screenshot-exact 29,898,425-parameter experimental architecture.

IMPORTANT: DDSE in this source is a custom two-gate recurrent state update,
NOT a full GRU/nn.GRUCell. It matches the screenshot parameter count only.
For mathematically exact GRUCells, redesign/recount the architecture.

No pretrained Zipformer/Conformer/Transformer backbone and no CTC or RNN-T.

IMPORTANT:
- This is a research prototype, NOT a trained Vietnamese ASR.
- The dynamic program assumes each target syllable occupies one contiguous
  segment, and the input is speech-trimmed. It does not yet model skip/silence.
- Batch examples must have equal unpadded feature lengths for this prototype.
- Targets are (initial_id, rhyme_id, tone_id), indexed independently.
- Inference uses a syllable lexicon to prevent invalid combinations.

Requires PyTorch 2.x. Test via: python dysyll_asr.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class DySyllConfig:
    n_mels: int = 80
    n_prosody: int = 4  # e.g. f0-normalized, voicing, energy, phonation
    hidden: int = 128
    phonetic_dim: int = 128
    prosodic_dim: int = 64
    n_initials: int = 24
    n_rhymes: int = 150
    n_tones: int = 6
    min_segment: int = 2  # frames AFTER 4x subsampling (80 ms at 10ms hop)
    max_segment: int = 20  # 800 ms at 10ms input hop


class AcousticResonanceFrontend(nn.Module):
    """Learned temporal filters and channel mixing, total temporal stride=4."""
    def __init__(self, cfg: DySyllConfig):
        super().__init__()
        d = cfg.hidden
        self.net = nn.Sequential(
            nn.Conv1d(cfg.n_mels, cfg.n_mels, 7, stride=2, padding=3,
                      groups=cfg.n_mels),
            nn.Conv1d(cfg.n_mels, d, 1), nn.SiLU(),
            nn.Conv1d(d, d, 5, stride=2, padding=2, groups=d),
            nn.Conv1d(d, d, 1), nn.SiLU(),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, mel: Tensor) -> Tensor:
        # mel: [batch, input_frames, mel_bins]
        return self.norm(self.net(mel.transpose(1, 2)).transpose(1, 2))


class DualDynamicsSyllableClock(nn.Module):
    """Two acoustically coupled memories and a learned, differentiable event clock.

    The phonetic memory is relatively fast; the prosodic memory receives
    acoustic evidence, voicing/pitch/phonation, and feedback from phonetics.
    Event hazard influences how the syllable clock is softly reset.
    """
    def __init__(self, cfg: DySyllConfig):
        super().__init__()
        H, P, D = cfg.phonetic_dim, cfg.prosodic_dim, cfg.hidden
        self.pho_gate = nn.Linear(D + H + P, H)
        self.pho_candidate = nn.Linear(D + H + P, H)
        self.pros_gate = nn.Linear(cfg.n_prosody + D + H + P, P)
        self.pros_candidate = nn.Linear(cfg.n_prosody + D + H + P, P)
        self.clock_rate = nn.Linear(D + H + P + 1, 1)
        self.boundary = nn.Linear(D + H + P + 2, 1)
        nn.init.constant_(self.boundary.bias, -2.0)

    def forward(self, acoustic: Tensor, prosody: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        B, T, _ = acoustic.shape
        H = self.pho_gate.out_features
        P = self.pros_gate.out_features
        h = acoustic.new_zeros((B, H))
        q = acoustic.new_zeros((B, P))
        phase = acoustic.new_zeros((B, 1))
        last_x = acoustic.new_zeros((B, acoustic.size(-1)))
        hs, qs, bs, phases = [], [], [], []
        for t in range(T):
            x = acoustic[:, t]
            pitch = prosody[:, t]
            h_input = torch.cat([x, h, q], dim=-1)
            z_h = torch.sigmoid(self.pho_gate(h_input))
            h = (1 - z_h) * h + z_h * torch.tanh(self.pho_candidate(h_input))
            q_input = torch.cat([pitch, x, h, q], dim=-1)
            z_q = torch.sigmoid(self.pros_gate(q_input))
            q = (1 - z_q) * q + z_q * torch.tanh(self.pros_candidate(q_input))
            rate_input = torch.cat([x, h, q, phase], dim=-1)
            increment = 0.04 + 0.40 * torch.sigmoid(self.clock_rate(rate_input))
            phase_before_reset = phase + increment
            spectral_flux = (x - last_x).abs().mean(dim=-1, keepdim=True)
            b_input = torch.cat([x, h, q, phase_before_reset, spectral_flux], dim=-1)
            boundary_hazard = torch.sigmoid(self.boundary(b_input))
            # Expected phase after a soft event; gradients reach both modules.
            phase = (1.0 - boundary_hazard) * phase_before_reset
            hs.append(h)
            qs.append(q)
            bs.append(boundary_hazard.squeeze(-1))
            phases.append(phase.squeeze(-1))
            last_x = x
        return (torch.stack(hs, 1), torch.stack(qs, 1),
                torch.stack(bs, 1), torch.stack(phases, 1))


class EventMemoryCompressor(nn.Module):
    """Scores every possible short contiguous segment (end_t, duration)."""
    def __init__(self, cfg: DySyllConfig):
        super().__init__()
        self.cfg = cfg
        H, P = cfg.phonetic_dim, cfg.prosodic_dim
        self.compress = nn.Sequential(
            nn.Linear(2 * (H + P) + 1, cfg.hidden),
            nn.SiLU(), nn.LayerNorm(cfg.hidden),
        )
        self.initial_head = nn.Linear(cfg.hidden, cfg.n_initials)
        self.rhyme_head = nn.Linear(cfg.hidden, cfg.n_rhymes)
        self.tone_head = nn.Linear(cfg.hidden, cfg.n_tones)

    def forward(self, h: Tensor, q: Tensor, boundary_hazard: Tensor) -> Dict[str, Tensor]:
        B, T, H = h.shape
        P, D = q.shape[-1], self.cfg.max_segment
        dev = h.device
        ends = torch.arange(1, T + 1, device=dev)[:, None].expand(T, D)
        durations = torch.arange(1, D + 1, device=dev)[None, :].expand(T, D)
        starts = (ends - durations).clamp(min=0)
        valid = (ends >= durations) & (durations >= self.cfg.min_segment)

        p_h = torch.cat((h.new_zeros(B, 1, H), torch.cumsum(h, dim=1)), dim=1)
        p_q = torch.cat((q.new_zeros(B, 1, P), torch.cumsum(q, dim=1)), dim=1)
        mean_h = (p_h[:, ends] - p_h[:, starts]) / durations[None, :, :, None]
        mean_q = (p_q[:, ends] - p_q[:, starts]) / durations[None, :, :, None]
        end_h = h[:, ends - 1]
        end_q = q[:, ends - 1]
        dur_feature = (durations.float().log1p() / 4)[None, :, :, None].expand(B, -1, -1, -1)
        compressed = self.compress(torch.cat([mean_h, end_h, mean_q, end_q, dur_feature], -1))

        # Bernoulli event prior: there is a boundary at each segment end and
        # no boundary strictly inside the proposed segment.
        # A forced final boundary ensures all utterances terminate.
        hazard = torch.cat([boundary_hazard[:, :-1], torch.ones_like(boundary_hazard[:, -1:])], dim=1)
        log_yes = hazard.clamp(min=1e-6).log()
        log_no = torch.log1p(-hazard.clamp(max=1 - 1e-6))
        prefix_log_no = torch.cat([log_no.new_zeros(B, 1), log_no.cumsum(dim=1)], dim=1)
        log_prior = log_yes[:, ends - 1] + prefix_log_no[:, ends - 1] - prefix_log_no[:, starts]
        log_prior = log_prior.masked_fill(~valid[None, :, :], -1e7)
        return {
            'initial': F.log_softmax(self.initial_head(compressed), dim=-1),
            'rhyme': F.log_softmax(self.rhyme_head(compressed), dim=-1),
            'tone': F.log_softmax(self.tone_head(compressed), dim=-1),
            'segment_logprior': log_prior,
            'valid_segments': valid,
            'boundary_hazard': boundary_hazard,
        }


class DySyllASR(nn.Module):
    def __init__(self, cfg: DySyllConfig | None = None):
        super().__init__()
        self.cfg = cfg or DySyllConfig()
        self.frontend = AcousticResonanceFrontend(self.cfg)
        self.dynamics = DualDynamicsSyllableClock(self.cfg)
        self.compressor = EventMemoryCompressor(self.cfg)

    def forward(self, mel: Tensor, prosody: Tensor) -> Dict[str, Tensor]:
        """mel [B, T, 80]; prosody [B, T, 4]. No input padding in v0.1."""
        if mel.ndim != 3 or prosody.ndim != 3 or mel.shape[:2] != prosody.shape[:2]:
            raise ValueError('mel and prosody must be [B, same_T, features]')
        if mel.shape[-1] != self.cfg.n_mels or prosody.shape[-1] != self.cfg.n_prosody:
            raise ValueError('Incorrect mel/prosody feature dimension')
        a = self.frontend(mel)
        p = F.avg_pool1d(prosody.transpose(1, 2), kernel_size=4, stride=4,
                         ceil_mode=True).transpose(1, 2)
        if p.size(1) != a.size(1):
            p = F.interpolate(p.transpose(1, 2), size=a.size(1), mode='nearest').transpose(1, 2)
        h, q, b, phase = self.dynamics(a, p)
        out = self.compressor(h, q, b)
        out['syllable_clock_phase'] = phase
        return out

    def segmental_nll(self, out: Mapping[str, Tensor], target: Tensor) -> Tensor:
        """Differentiable exact log-marginal over target-compatible segmentations.

        Args:
            out: forward output, with batch=1.
            target: [K,3] integer initial/rhyme/tone ids.
        Returns: scalar -log P(target|speech).

        Not CTC: one multi-attribute syllable is emitted per latent segment.
        Note: this v0.1 implementation assumes no silence segments and one
        target syllable per segment. Trim silence before computing this loss.
        """
        scores = out['segment_logprior']
        if scores.size(0) != 1 or target.ndim != 2 or target.shape[-1] != 3:
            raise ValueError('v0.1 segmental_nll requires batch=1 and target [K,3]')
        T, D = scores.shape[1:]
        K = target.size(0)
        if K < 1 or K * self.cfg.min_segment > T or K * D < T:
            raise ValueError('Target syllable count incompatible with segment lengths')
        neg = scores.new_tensor(-1e7)
        zero = scores.new_tensor(0.0)

        # Denominator log Z(x): all segmentations allowed by min/max duration.
        z: List[Tensor] = [zero]
        for t in range(1, T + 1):
            options = [z[t - dur] + scores[0, t - 1, dur - 1]
                       for dur in range(self.cfg.min_segment, min(D, t) + 1)]
            z.append(torch.logsumexp(torch.stack(options), 0) if options else neg)

        # Numerator: exactly K segments. Vectorize over all possible end
        # frames and durations; loop only over K output syllables.
        ends = torch.arange(1, T + 1, device=scores.device)[:, None]
        durations = torch.arange(1, D + 1, device=scores.device)[None, :]
        starts = (ends - durations).clamp(min=0)
        prev = torch.cat([zero[None], neg.expand(T)])
        for k in range(K):
            i_id, r_id, t_id = [int(u) for u in target[k].tolist()]
            label_score = (out['initial'][0, :, :, i_id]
                           + out['rhyme'][0, :, :, r_id]
                           + out['tone'][0, :, :, t_id])
            updated = torch.logsumexp(prev[starts] + scores[0] + label_score, dim=1)
            prev = torch.cat([neg[None], updated])
        return z[-1] - prev[-1]

    @torch.no_grad()
    def decode_one(self, out: Mapping[str, Tensor],
                   lexicon: Mapping[Tuple[int, int, int], str] | None = None) -> List[Tuple[int, int, int] | str]:
        """Viterbi event decoding, batch=1. If lexicon provided, output words.

        lexicon maps (initial, rhyme, tone) to Vietnamese syllable spellings.
        Without it outputs predicted triples, NOT guaranteed valid syllables.
        An external language model can be added to the search score later.
        """
        segment = out['segment_logprior']
        if segment.size(0) != 1:
            raise ValueError('decode_one expects batch=1')
        T, D = segment.shape[1:]
        if lexicon:
            candidates = torch.tensor(list(lexicon), dtype=torch.long, device=segment.device)
        else:
            candidates = None
        best = [0.0] + [-float('inf')] * T
        back = [None] * (T + 1)
        for end in range(1, T + 1):
            for dur in range(self.cfg.min_segment, min(D, end) + 1):
                start = end - dur
                if best[start] == -float('inf'):
                    continue
                lp_i = out['initial'][0, end-1, dur-1]
                lp_r = out['rhyme'][0, end-1, dur-1]
                lp_t = out['tone'][0, end-1, dur-1]
                if candidates is None:
                    triple = (int(lp_i.argmax()), int(lp_r.argmax()), int(lp_t.argmax()))
                    label_lp = lp_i[triple[0]] + lp_r[triple[1]] + lp_t[triple[2]]
                else:
                    cand_score = (lp_i[candidates[:, 0]] + lp_r[candidates[:, 1]]
                                  + lp_t[candidates[:, 2]])
                    idx = int(cand_score.argmax())
                    triple = tuple(map(int, candidates[idx].tolist()))
                    label_lp = cand_score[idx]
                val = best[start] + float(segment[0, end-1, dur-1] + label_lp)
                if val > best[end]:
                    best[end], back[end] = val, (start, triple)
        if back[T] is None:
            return []
        emitted: List[Tuple[int, int, int] | str] = []
        pos = T
        while pos > 0:
            start, triple = back[pos]
            emitted.append(lexicon[triple] if lexicon else triple)
            pos = start
        return emitted[::-1]


def quick_smoke_test() -> None:
    torch.manual_seed(7)
    cfg = DySyllConfig(hidden=64, phonetic_dim=48, prosodic_dim=24,
                       n_initials=24, n_rhymes=150, n_tones=6,
                       min_segment=2, max_segment=12)
    net = DySyllASR(cfg)
    mel = torch.randn(1, 64, cfg.n_mels)  # 16 frames after subsampling
    prosody = torch.randn(1, 64, cfg.n_prosody)
    target = torch.tensor([[2, 10, 1], [3, 22, 4], [9, 7, 0]], dtype=torch.long)
    output = net(mel, prosody)
    loss = net.segmental_nll(output, target)
    assert torch.isfinite(loss), f'Non-finite loss: {loss}'
    assert loss.item() >= -1e-4, 'NLL must not be negative'
    loss.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() and
                  p.grad.abs().sum().item() > 0 for p in net.parameters())
    assert grad_ok, 'No finite nonzero gradient'
    prediction = net.decode_one(output)
    print('Model: DySyll-ASR v0.1')
    print('Parameters:', sum(p.numel() for p in net.parameters()))
    print('Frontend output frames:', output['initial'].shape[1])
    print('Loss:', round(float(loss.item()), 4), 'backward: OK')
    print('Untrained token triples:', prediction)


if __name__ == '__main__':
    quick_smoke_test()
