# Scenarios — Full Reference

WhyDidItFail has 12 scenarios across three difficulty tiers. Each one plants a specific, realistic ML failure mode into a synthetic training run. The agent must read the available evidence and name the failure.

This document explains every scenario in detail: what the data looks like, why the failure happens, what the agent needs to see, and what a correct diagnosis and fix look like.

---

## How Scenarios Work

Every scenario is a Python dictionary with the same structure:

| Field | What it is |
|---|---|
| `difficulty` | `easy`, `medium`, or `hard` |
| `required_sources` | Which data sources the agent must inspect before submitting |
| `logs` | Per-epoch training and validation metrics |
| `config` | Hyperparameter configuration |
| `gradient_norms` | Per-layer or per-epoch gradient magnitudes |
| `correct_diagnosis` | The exact internal label the grader checks against |
| `correct_fix` | The expected remediation (used by the fix scorer) |

The `required_sources` field is the contract between the scenario and the grader. Inspecting a required source earns the agent reward. Submitting without inspecting one costs points. Inspecting a source not in the list is penalised mildly.

---

## Easy Tier — Logs Only

Easy scenarios are diagnosable from the training logs alone. The config and gradients exist in the data but are **not required** — inspecting them costs the agent −0.02 each.

The four easy scenarios exercise four distinct log patterns: catastrophic divergence (NaN), chaotic oscillation, train-val split, and flatness.

---

### Exploding Gradients

**Correct diagnosis:** `exploding_gradients`  
**Required sources:** logs  
**Correct fix:** `enable gradient clipping (clip_grad_norm=1.0)`

**What happens:**  
The model starts normally at epoch 1 (`train_loss=2.31`), then catastrophically diverges. By epoch 2, loss has jumped to `847.2`. By epoch 3, both train and validation loss are `NaN`. The gradient norms tell the same story: `0.43` at epoch 1, `6821.4` at epoch 2, then `NaN`.

The config shows a reasonable learning rate (`lr=0.001`, Adam optimizer) with `clip_grad=False`. The learning rate is not the culprit — the problem is that gradients are not being clipped and grow uncontrollably during backpropagation.

**What the agent needs to see:**  
The NaN in the logs is the definitive signal. Loss going from 2.31 → 847.2 → NaN in three epochs is an unambiguous divergence pattern. No config or gradient inspection is required to make this call.

**Why it's easy:**  
NaN loss is one of the most recognisable failure patterns in ML. There's no ambiguity, no competing hypothesis — any agent that reads the logs and sees NaN should label this correctly.

**Disambiguation note:**  
The config shows `lr=0.001` (not high). This rules out "learning rate too high". Once training has already produced finite loss at epoch 1, the failure at epochs 2-3 is exploding gradients, not bad weight initialization (which would be NaN from epoch 1).

---

### Learning Rate Too High

**Correct diagnosis:** `learning_rate_too_high`  
**Required sources:** logs  
**Correct fix:** `reduce learning_rate to 0.01`

**What happens:**  
The loss oscillates wildly with no convergence trend. It goes `2.30 → 15.21 → 0.82 → 9.73 → 3.15` across five epochs. There's no NaN — the model is technically alive — but it can't settle. Accuracy fluctuates similarly: `0.11 → 0.47 → 0.31 → 0.44 → 0.29`. The gradient norms swing between 0.19 and 11.2.

The config shows `lr=1.0` with SGD. A learning rate of 1.0 is far too large for SGD on CIFAR-10 with a VGG16. Each gradient step overshoots the loss minimum, sending the model into a different part of the landscape every epoch.

**What the agent needs to see:**  
The oscillation pattern is visible from the logs alone. The lr in the logs (`"lr": 1.0`) also appears inline with each epoch — a sharp agent can confirm the culprit without even inspecting the config separately.

**Why it's easy:**  
Wildly oscillating loss with no NaN is a classic high-LR symptom. The `lr=1.0` value in the log lines is a strong additional signal.

**Disambiguation note:**  
This scenario uses `batch_size=64`, which rules out batch_size_too_small (which has `batch_size=2`). The oscillation here is caused by the step size, not by gradient variance from small batches.

---

### Overfitting

**Correct diagnosis:** `overfitting`  
**Required sources:** logs  
**Correct fix:** `increase dropout to 0.3 and weight_decay to 0.01`

**What happens:**  
Train loss falls steadily while validation loss climbs: `train_loss` goes from `2.10 → 0.06` by epoch 20, while `val_loss` rises from `2.16 → 2.74`. Train accuracy reaches `0.99`; val accuracy falls from `0.55` to `0.49`. The model has memorised the training data and fails to generalise.

The config shows `weight_decay=0.001` and `dropout=0.1` — regularisation is present but insufficient for a ResNet50 on CIFAR-10. The model is powerful enough to overfit even with light regularisation.

**What the agent needs to see:**  
The train-val divergence pattern in the logs is the definitive signal. `train_acc=0.99` and `val_acc=0.49` by epoch 20 is a textbook generalisation gap.

**Why it's easy:**  
Train-val divergence is one of the most studied phenomena in ML. The logs make it immediately visible.

**Disambiguation note:**  
The config shows **non-zero** `weight_decay=0.001` and `dropout=0.1`. This distinguishes it from `missing_regularization`, which has `weight_decay=0.0` and `dropout=0.0`. An agent that reads the log pattern correctly but then inspects the config will see regularisation present — the correct label remains "overfitting" because the regularisation is insufficient, not absent.

---

### Underfitting

**Correct diagnosis:** `underfitting`  
**Required sources:** logs  
**Correct fix:** `increase model capacity or use a deeper architecture`

**What happens:**  
Both train and validation loss stay high with almost no improvement over 20 epochs. `train_loss` goes from `2.29 → 2.21`. `train_acc` barely moves: `0.11 → 0.14`. Val metrics track train metrics almost exactly — there is no train-val gap. The model is stuck near random baseline (~10% on 10-class CIFAR-10).

The config reveals the cause: `architecture=LinearClassifier`. A linear model simply doesn't have the representational capacity to learn CIFAR-10. It's the wrong tool for the job.

**What the agent needs to see:**  
Both losses are high and similar — there's no generalisation gap. This is the defining characteristic of underfitting. The architecture field in the config confirms it, but the log pattern alone is sufficient.

**Why it's easy:**  
Flat losses near random baseline are unambiguous. There's no NaN, no oscillation, no divergence — just failure to learn.

**Disambiguation note:**  
The config shows `optimizer=adam`, `momentum` not applicable. This rules out `optimizer_misconfiguration` (which requires SGD with momentum=0.0). If the agent is careful, it notes that `weight_decay=0.0` — but that's irrelevant here because both losses are high and similar, which points to capacity problems, not regularisation.

---

## Medium Tier — Logs + Config

Medium scenarios cannot be diagnosed from logs alone. The log pattern is ambiguous — it looks like something, but the config is needed to confirm or redirect the diagnosis. Both sources are required.

---

### Learning Rate Too Low

**Correct diagnosis:** `learning_rate_too_low`  
**Required sources:** logs, config  
**Correct fix:** `increase learning_rate to 0.001`

**What happens:**  
Loss is decreasing, but imperceptibly. Over 20 epochs: `2.302 → 2.298 → 2.290 → 2.275`. The model is technically learning — loss goes down — but at a rate that would take thousands of epochs to reach convergence. Gradient norms are tiny (`0.0031 → 0.0021`) but non-zero.

The config reveals the cause: `lr=0.000001` (1e-6). Adam is being used, but with a learning rate 1000× smaller than typical. Each update moves the weights by almost nothing.

**What the agent needs to see:**  
The logs alone show slow convergence, but "slow convergence" could also be underfitting (wrong architecture) or a hard dataset. The config's `lr=0.000001` is the confirmation — this tiny value is the explicit reason for the glacial pace.

**Why it requires both sources:**  
From the logs alone, the agent sees: loss is decreasing but slowly. That's ambiguous. Only when the config shows `lr=0.000001` does the diagnosis become unambiguous — the optimizer is working correctly, just with too small a step size.

**Disambiguation note:**  
Gradient norms are small but non-zero (0.003 range). This rules out vanishing gradients, where norms would be exponentially decaying toward zero across layers. The norms are uniformly small here because the LR makes every update tiny, not because gradients are being crushed during backprop.

---

### Missing Regularization

**Correct diagnosis:** `missing_regularization`  
**Required sources:** logs, config  
**Correct fix:** `add weight_decay=0.01 and dropout=0.3`

**What happens:**  
Train loss drops fast and val loss rises: `train_loss=0.01`, `val_loss=2.10` by epoch 30. `train_acc=1.00`, `val_acc=0.56`. The log pattern looks exactly like overfitting.

But the config shows `weight_decay=0.0` and `dropout=0.0`. There is no regularisation at all. A ResNet101 on CIFAR-10 with zero regularisation will memorise the training set completely — not because the regularisation is insufficient, but because it was never added.

**What the agent needs to see:**  
The log pattern alone is indistinguishable from `overfitting`. Only the config confirms the distinction: `overfitting` has `weight_decay=0.001, dropout=0.1` (regularisation present but insufficient), while `missing_regularization` has both at exactly `0.0`.

**Why it requires both sources:**  
Logs → "train-val divergence, probable overfitting". Config → "weight_decay=0.0 AND dropout=0.0" → "missing regularization, not overfitting". The config is the discriminating evidence.

**Disambiguation note:**  
The label distinction matters for the fix. "Overfitting" → increase existing regularisation. "Missing regularization" → add regularisation from scratch. An agent that labels this "overfitting" has diagnosed the symptom correctly but missed the root cause.

---

### Batch Size Too Small

**Correct diagnosis:** `batch_size_too_small`  
**Required sources:** logs, config  
**Correct fix:** `increase batch_size to at least 32`

**What happens:**  
Loss oscillates wildly across 8 epochs: `4.12 → 3.87 → 4.45 → 3.21 → 4.78 → 3.44 → 4.93 → 3.67`. There's a downward trend on average, but every epoch spikes or drops dramatically. Gradient norms alternate: `1.21 → 0.38 → 2.04 → 0.29 → 1.87 → 0.41 → 2.31 → 0.55`.

The config reveals the cause: `batch_size=2`. With only 2 samples per gradient update, each update is a noisy estimate of the true gradient. The model steps in a random direction every epoch — sometimes helpful, sometimes not.

**What the agent needs to see:**  
The oscillation pattern in the logs looks like "learning rate too high" — and the agent might guess that first. The config's `batch_size=2` is the discriminating signal. The learning rate is reasonable (`lr=0.001` with SGD + momentum=0.9).

**Why it requires both sources:**  
Oscillating loss is caused by both high LR and tiny batch sizes. The logs alone cannot disambiguate. Only after seeing `batch_size=2` in the config can the agent correctly attribute the noise to gradient variance rather than step size.

**Disambiguation note:**  
The learning rate (`lr=0.001`) and optimizer (`sgd`, `momentum=0.9`) are standard — this is not a high LR situation. The batch size is the anomaly. An agent that stops at the logs and labels this "learning rate too high" has confused the symptom with the wrong cause.

---

### Optimizer Misconfiguration

**Correct diagnosis:** `optimizer_misconfiguration`  
**Required sources:** logs, config  
**Correct fix:** `set momentum=0.9 for SGD optimizer`

**What happens:**  
Both train and validation loss decline extremely slowly and plateau: `2.30 → 2.25 → 2.25 → 2.23 → 2.22` over 20 epochs. The losses are similar (no train-val gap), and accuracy never improves meaningfully. Gradient norms are normal (`0.42 → 0.36`), ruling out gradient-level pathologies.

The config shows the cause: `optimizer=sgd, momentum=0.0`. SGD with no momentum has no gradient averaging. On a complex loss landscape with saddle points and flat regions, momentum is what allows SGD to keep moving. Without it, the optimizer stalls on flat plateaus and saddle points.

**What the agent needs to see:**  
The logs show flat losses similar to underfitting. The config reveals `momentum=0.0` as the specific misconfiguration — this is not an architecture capacity problem, it's a missing optimizer hyperparameter.

**Why it requires both sources:**  
From logs alone, flat losses near baseline look identical to underfitting (wrong architecture). Only the config — specifically `optimizer=sgd, momentum=0.0` — reveals the true cause.

**Disambiguation note:**  
The architecture is `ResNet18`, which is not a capacity-limited model like `LinearClassifier` in the underfitting scenario. The learning rate (`lr=0.01`) is also reasonable. The only anomaly is `momentum=0.0`. An agent that sees flat losses and labels this "underfitting" has missed the config evidence.

---

## Hard Tier — Logs + Config + Gradients

Hard scenarios require all three data sources. The logs alone are ambiguous, the config narrows it down, and the gradient norms provide the final confirming evidence. All three are required.

---

### Vanishing Gradients

**Correct diagnosis:** `vanishing_gradients`  
**Required sources:** logs, config, gradients  
**Correct fix:** `switch activation to relu and add batch normalization`

**What happens:**  
Training on MNIST with a 20-layer MLP using sigmoid activations. Loss barely moves over 20 epochs: `2.30 → 2.27`, accuracy stuck at `0.11-0.12`. The logs suggest the model is barely learning.

The config reveals the activation: `activation=sigmoid`. The gradients tell the full story: the output layer has norm `0.21`, but by layer 1 (the input end) the norm has decayed to `0.00000001` — eight orders of magnitude smaller. The gradient is effectively zero at the input layers, so those weights never update.

**What the agent needs to see:**  
Logs → slow/stuck learning (ambiguous). Config → sigmoid activation in a deep network (suspicious). Gradients → exponential decay from output to input (`0.21 → 0.0031 → 0.000042 → 0.0000003 → 0.00000001`) confirms vanishing gradients. All three required.

**The mechanism:**  
Sigmoid squashes its output to (0, 1) and its gradient is at most 0.25. In a 20-layer network, the gradient is multiplied by this factor at every layer during backpropagation. `0.25^20 ≈ 10^-12` — essentially zero by the time it reaches the input.

**Disambiguation note:**  
The gradient decay is exponential across layers, and each layer's norm is non-zero (unlike dying ReLU, where norms are exactly 0.0). The config explicitly states `activation=sigmoid`, which is the known culprit for this failure mode.

---

### Dying ReLU

**Correct diagnosis:** `dying_relu`  
**Required sources:** logs, config, gradients  
**Correct fix:** `reduce learning_rate to 0.001 or switch to leaky_relu activation`

**What happens:**  
Training starts: `train_loss=2.31` at epoch 1, then partially improves to `1.95` by epoch 2. But then it freezes — epochs 2, 3, and 5 all show identical numbers: `train_loss=1.95, val_loss=2.01, train_acc=0.28`. The model has stopped learning completely.

The config shows `lr=0.1` (high for SGD) and `activation=relu`. The gradients explain the freeze: the output layer has norm `0.15`, but every hidden layer (`layer_8`, `layer_6`, `layer_4`, `layer_2`) shows a gradient norm of exactly `0.0`. Not small — exactly zero.

**What the agent needs to see:**  
Logs → partial improvement then sudden freeze (unusual pattern). Config → high LR with ReLU (combination that causes dying ReLU). Gradients → exact zeros in all hidden layers confirms dead neurons.

**The mechanism:**  
ReLU neurons output 0 when their input is negative. With a high learning rate, a large gradient update can push a neuron's weights so negative that its input is always negative — making it always output 0. Once dead, the gradient through a dead ReLU is also 0, so the neuron can never recover. At `lr=0.1`, many neurons die in the first few updates, causing the training loss to freeze.

**Disambiguation note:**  
The key signal is gradient norms of exactly `0.0` in hidden layers. Vanishing gradients produce tiny but nonzero norms (e.g. `1e-8`). Dying ReLU produces norms that are precisely zero because the ReLU derivative is a hard 0 for dead neurons — not an approximation.

---

### Bad Weight Initialization

**Correct diagnosis:** `bad_weight_initialization`  
**Required sources:** logs, config, gradients  
**Correct fix:** `use kaiming or xavier weight initialization`

**What happens:**  
Loss is `NaN` from epoch 1. There's no "first good epoch" — the model fails immediately. The gradient norms are astronomically large: `98432.1`, `74219.8`, `55103.4` across the first three layers. These values are orders of magnitude higher than any normal run.

The config shows `weight_init=normal_std_100`. Initialising weights from a normal distribution with standard deviation 100 means the initial weights are enormous. The first forward pass immediately overflows: large weights → enormous pre-activations → NaN in the loss.

**What the agent needs to see:**  
Logs → NaN from epoch 1 (but NaN from epoch 1 could also be exploding gradients in a different sense). Config → `weight_init=normal_std_100` (the explicit anomaly). Gradients → norms in the tens of thousands confirming the extreme magnitude.

**Why it's different from exploding gradients:**  
`exploding_gradients` shows a finite loss at epoch 1 that then diverges. `bad_weight_initialization` is NaN from the very first epoch, before training has meaningfully begun. The config's `weight_init=normal_std_100` is the discriminating evidence. The correct rule: if NaN starts at epoch 1 AND config shows extreme weight_init → bad weight initialization. If NaN starts after epoch 1 → exploding gradients.

**Disambiguation note:**  
An agent that labels this "exploding gradients" based on the NaN alone has not used the config or gradient evidence. The gradient norms (`98432`) and weight_init (`std=100`) are specific, confirming evidence that changes the label.

---

### LR Scheduler Misconfiguration

**Correct diagnosis:** `lr_scheduler_misconfiguration`  
**Required sources:** logs, config, gradients  
**Correct fix:** `set gamma to 0.1 so the scheduler decreases lr instead of increasing it`

**What happens:**  
Training progresses normally for the first few epochs, then suddenly explodes at epoch 6: `train_loss` jumps from `1.20 → 9.87`. The model recovers and trains again, then explodes again at epoch 11: `train_loss=87.3`. The log shows the lr inline: `lr=0.001` at epoch 5, `lr=0.01` at epoch 6, `lr=0.1` at epoch 11 — the learning rate is growing 10× every 5 epochs. Gradient norms follow: `0.42` at epoch 5, `18.73` at epoch 6, `156.2` at epoch 11.

The config reveals the cause: `lr_scheduler=StepLR, step_size=5, gamma=10.0`. StepLR multiplies the learning rate by `gamma` every `step_size` epochs. With `gamma=10.0` (a value greater than 1.0), the scheduler is **increasing** the learning rate at each step instead of decreasing it. This is a configuration error — the intended behaviour is `gamma < 1.0` (e.g. 0.1) to decay the learning rate over time.

**What the agent needs to see:**  
Logs → periodic spikes at epochs 6 and 11, correlated with lr jumps. Config → `StepLR, gamma=10.0` (the `> 1.0` gamma is the bug). Gradients → norm spikes at the same intervals confirming the lr jump as the trigger.

**The mechanism:**  
StepLR is a common scheduler used to reduce the learning rate as training progresses (e.g. every 5 epochs, multiply by 0.1). Setting `gamma=10.0` inverts this — every 5 epochs the lr grows by 10×. By epoch 11, the lr has gone from 0.001 to 0.1, which at that point is high enough to cause divergence.

**Disambiguation note:**  
The loss spikes are **periodic and predictable** — exactly every 5 epochs. This is the distinguishing characteristic of a scheduler bug. Random exploding gradients don't follow a fixed interval. An agent that sees the spike at epoch 6 and doesn't look for another one at epoch 11 may miss the periodic pattern.

---

## Scenario Matrix

| Scenario | Tier | Required Sources | Key Signal | Discriminator |
|---|---|---|---|---|
| exploding_gradients | Easy | logs | loss → NaN after epoch 1 | NaN after finite start |
| learning_rate_too_high | Easy | logs | loss oscillates wildly, stays finite | lr=1.0 visible in logs |
| overfitting | Easy | logs | train-val gap widens | val_loss rising while train_loss falls |
| underfitting | Easy | logs | both losses flat near baseline | no train-val gap |
| learning_rate_too_low | Medium | logs + config | imperceptibly slow loss decrease | config: lr=1e-6 |
| missing_regularization | Medium | logs + config | train-val divergence (same as overfitting) | config: weight_decay=0.0 AND dropout=0.0 |
| batch_size_too_small | Medium | logs + config | oscillating loss | config: batch_size=2 |
| optimizer_misconfiguration | Medium | logs + config | flat losses (same as underfitting) | config: sgd, momentum=0.0 |
| vanishing_gradients | Hard | logs + config + gradients | slow learning | gradients decay exponentially by layer |
| dying_relu | Hard | logs + config + gradients | partial improvement then frozen | gradients exactly 0.0 in hidden layers |
| bad_weight_initialization | Hard | logs + config + gradients | NaN from epoch 1 | config: weight_init=normal_std_100 |
| lr_scheduler_misconfiguration | Hard | logs + config + gradients | periodic loss spikes | config: gamma=10.0 (> 1.0) |

---

## Design Principles

**Paired ambiguity:** Several scenarios are deliberately paired to test fine-grained discrimination:
- `overfitting` vs `missing_regularization` — same log pattern, different config
- `underfitting` vs `optimizer_misconfiguration` — same log pattern, different config
- `exploding_gradients` vs `bad_weight_initialization` — both NaN, different timing
- `vanishing_gradients` vs `dying_relu` — both near-zero gradients, different kind (decay vs exact zero)

**Progressive evidence:** Each tier adds one more required source. Easy = logs only. Medium = logs must be correlated with config. Hard = all three sources must align. An agent that inspects only logs on a hard scenario will have ambiguous evidence and is likely to mis-diagnose.

**Realistic data:** The numeric values (loss curves, gradient magnitudes, learning rates) are calibrated to match what a practitioner would actually see in a real training run, not toy examples.