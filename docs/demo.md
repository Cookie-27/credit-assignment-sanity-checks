# Captured CLI demonstration

These are actual outputs captured by `python scripts/build_demo.py`.
The accompanying GIF renders the same transcript, with five seconds per case;
its playback timing does not represent program runtime.

Run all commands from the repository root. See the [walkthrough](walkthrough.md)
for Python examples and [research note](research_note.md) for assumptions.

## 1. A proxy can reverse the update direction

```bash
python -m credit_sanity demo
```

```text
Expected reward: 0.999000000
True dJ/dtheta: +0.000900000000
method                                expected gradient    reward change (step=.001)
grpo                                    +0.002377806524          +2.140023835917e-09
trajectory_raw                          +0.025626506224          +2.306361918336e-08
trajectory_smooth                       -0.180982441841          -1.628959897393e-07
trajectory_raw_unprotected              +0.211413513010          +1.902560719191e-07
trajectory_smooth_unprotected           -0.201619132391          -1.814718538162e-07
Toy first-update diagnostic; not an LLM training result.
```

The true gradient is positive; the smoothed proxy is negative in this case.

## 2. Equal-value drift need not reduce reward

```bash
python -m credit_sanity demo --q0 1 --q1 1
```

```text
Expected reward: 1.000000000
True dJ/dtheta: +0.000000000000
method                                expected gradient    reward change (step=.001)
grpo                                    +0.000000000000          -0.000000000000e+00
trajectory_raw                          +0.020949198305          +0.000000000000e+00
trajectory_smooth                       -0.188541731061          -0.000000000000e+00
trajectory_raw_unprotected              +0.209491983047          +0.000000000000e+00
trajectory_smooth_unprotected           -0.209490812290          -0.000000000000e+00
Toy first-update diagnostic; not an LLM training result.
```

Both actions always succeed. Every displayed reward change is zero.

## 3. Complete action echo changes the result

```bash
python -m credit_sanity demo --echo
```

```text
Expected reward: 0.999000000
True dJ/dtheta: +0.000900000000
method                                expected gradient    reward change (step=.001)
grpo                                    +0.002377806524          +2.140023835917e-09
trajectory_raw                          +0.004755613045          +4.280043599536e-09
trajectory_smooth                       +0.004755613047          +4.280043600646e-09
trajectory_raw_unprotected              +0.004755613045          +4.280043599536e-09
trajectory_smooth_unprotected           +0.004755613047          +4.280043600646e-09
Toy first-update diagnostic; not an LLM training result.
```

Revealing the action makes the smoothed proxy positive in this case.

## 4. A ranking swap can preserve the total coefficient

```bash
python -m credit_sanity coefficient-demo
```

```text
{
  "before": [
    0.0,
    0.9999990000010001
  ],
  "after": [
    0.9999990000010001,
    0.0
  ],
  "sum_before": 0.9999990000010001,
  "sum_after": 0.9999990000010001,
  "assumption": "The two samples have identical policy inputs and target tokens.",
  "scope": "Coefficient identity only; this does not equate distinct neural-network gradients."
}
```

This identity requires identical policy inputs and target tokens.

## Rebuild the presentation

Text capture uses only the standard library. GIF authoring optionally needs
Pillow and a local monospace TrueType font:

```bash
python scripts/build_demo.py --gif --font /path/to/monospace.ttf
```

Pillow is not a dependency of the experiment, its tests, or the walkthrough.
