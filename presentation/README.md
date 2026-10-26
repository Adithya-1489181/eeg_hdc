# EEG-HDC explained from scratch

I've written this in the same order as the slides: the problem, then the ML basics you need, then the architecture, then the results.

---

## 1. The problem

**EEG** is a recording of the brain's electrical activity. You stick electrodes on the scalp and record voltage over time. Our recordings have **18 channels**, which means 18 wiggly lines at once. We sample **256 times per second**.

During a **seizure**, the brain's electrical activity becomes abnormal. Doctors want two answers:
1. **Detection:** is this stretch of EEG a seizure or not?
2. **Subtype:** if it is a seizure, which kind? We handle three:
   - **FA** (Focal Aware): the person stays conscious.
   - **FIA** (Focal Impaired Awareness): the person loses awareness.
   - **FBTC** (Focal to Bilateral Tonic-Clonic): starts in one area and spreads to the whole brain, with convulsions.

The type matters because treatment and surgery planning depend on it. Doing this by eye takes an expert hours, so we train a computer to do it.

---

## 2. Machine learning in 5 minutes

**Model:** a big mathematical function. You give it numbers (the EEG) and it gives you numbers back (for example "92% seizure").

**Weights (parameters):** the thousands of adjustable knobs inside the function. Our backbone has 445,697 of them.

**Training:** start with random knobs, show the model an example, and compare its answer with the true label. The gap is the **loss**. Nudge the knobs to shrink the loss, then repeat over thousands of examples. Eventually the knobs are set so the model gets answers right.

**Neural network:** a model built from layers. Each layer takes numbers in, does simple maths, and passes numbers on. Early layers pick up simple patterns and later layers combine them into complex ones. For example, edges lead to eyes, which lead to a face.

---

## 3. The building blocks of our model (the CRNN)

CRNN = **C**onvolutional **R**ecurrent **N**eural **N**etwork. It has three parts.

### (a) Convolution layers (the "CNN" part): spotting shapes
Think of a small sliding window that moves along the signal and asks "does this snippet look like a spike or a rhythm?" Each window is a **filter**. We stack 4 of these layers:
- Layer 1 uses a big filter (63 samples wide) and sees slow waves.
- Later layers use smaller filters (31, 15, 7) and see sharp details.
- Each layer also shrinks the signal in time (stride 2). After 4 layers the signal is **16 times shorter**, but it is described by more "feature channels" (32, 64, 128, 128).

Think of it as summarising a long, noisy signal into a short list of "what happened here".

### (b) Bi-LSTM (the "recurrent" part): remembering order
A seizure is a **pattern over time**: it starts, evolves and ends. An LSTM reads the sequence step by step and keeps a memory of what came before. **Bi**-directional means it reads forwards and also backwards, so every moment gets context from both past and future.

### (c) Attention pooling: deciding what matters
After the LSTM we have one vector per time step, but we want **one summary** per window. A plain average would let boring moments drown out the seizure moment. Attention gives each time step an importance score and takes a weighted average:

- `u_t = tanh(W·h_t + b)`: a small network scores each time step t.
- `α_t = exp(score_t) / Σ exp(score_τ)`: this is **softmax**. It turns scores into percentages that add up to 100%.
- `e = Σ α_t · h_t`: the final summary is the weighted sum.

The result **e** is a list of **128 numbers**. You can read it as the model's fingerprint of this EEG window. We call this an **embedding**.

> The whole chain (conv → LSTM → attention) is called the **backbone**. Its job is only to turn raw EEG into a good 128-number fingerprint. It doesn't make the final decision.

---

## 4. Key idea 1: Transfer learning, and "frozen" vs "fine-tuned"

**Problem:** we have lots of data for *seizure vs not seizure* (928,154 windows) but very little for *subtypes* (1,824 windows, from only 13 patients).

**Idea:** first teach a model the easy, data-rich task (detection). It learns general seizure patterns. Then reuse what it learned for the harder, data-poor task. That is **transfer learning**, like learning to read before studying medicine.

- **Phase A:** train the backbone to detect seizures on all the data.
- **Phase B:** copy that backbone and keep training it on subtype labels. This is **fine-tuning**: we start from good knobs and adjust them slightly. We lock the first 2 conv layers and let the rest adapt.

**Frozen** means the knobs are locked and never change. The backbone is only used to produce fingerprints. **Fine-tuned** means the knobs were allowed to adjust for a new task.

---

## 5. Key idea 2: Why two backbones? (the v4 model)

We found each backbone has a different strength:
- The **Phase A (detection) backbone** is good at spotting FA seizures but cannot tell FIA from FBTC.
- The **fine-tuned (subtype) backbone** separates FBTC, but fine-tuning on so little data damaged its ability to see FA.

So we use **both** and join their fingerprints end to end (**concatenation**):

> **z = [ e_frozen ‖ e_fine-tuned ] ∈ R²⁵⁶**

In words: take the 128 numbers from backbone 1 and the 128 numbers from backbone 2, and write them as one list of 256 numbers. The symbol `‖` means "stick together", and `R²⁵⁶` means "a list of 256 real numbers". Nothing mysterious is happening. We just give the final decision-maker two experts' opinions at once.

---

## 6. Key idea 3: The imbalance problem and SMOTE

In the subtype training data:
- FIA: **1,158** windows
- FBTC: **87**
- FA: **29**

A lazy model could always say "FIA" and still be right about 90% of the time, while being useless for the rare types.

**SMOTE** creates fake extra examples of the rare classes. Take a real rare example, find a similar neighbour, and create a new point somewhere on the line between them:

> `z_new = z_i + λ · (z_neighbour − z_i)`, where λ is a random number between 0 and 1.

If λ=0 you get the original, if λ=1 you get the neighbour, and if λ=0.5 you get the midpoint.

Two details matter:
- **We do this on the 256-number fingerprints, not on raw EEG.** Averaging two raw brain signals would create signals that no brain could produce. Averaging fingerprints is much safer.
- **Only on training data.** The test data stays 100% real, otherwise we'd be grading ourselves on fake data.

We boost FA 3× (29 → 87) and FBTC 5× (87 → 435). FIA is left alone.

---

## 7. The final decision-maker: HistGradientBoosting (HistGB)

The fingerprints go into a classifier. We tried three:
- **MLP:** a small neural network.
- **Random Forest:** many decision trees voting.
- **HistGB:** many small decision trees built one after another, where each new tree fixes the mistakes of the previous ones. We use 200 trees of depth 4.

HistGB won. MLP and Random Forest completely failed on FA (recall 0).

---

## 8. Key idea 4: The adversarial control (catching a cheating model)

This is the cleverest part of the project.

**The trap:** in our data, the subtype label belongs to the **patient**, not the individual seizure. Every seizure from patient PN00 has the same label. A model could "cheat" by recognising whose brain it is, from their personal EEG style, and then outputting that patient's known label. It would look accurate while understanding nothing about seizures.

**The test:** take normal, non-seizure EEG from each patient, stick that patient's subtype label on it, and run the same model. If the model scores well on this fake task, it is recognising the patient, not the seizure.

**The gap:** `Δ = recall on real seizures − recall on control`
- Δ > 0 means the model really uses seizure patterns.
- Δ ≤ 0 means it is probably just identifying the patient.

Random Forest scored Δ = −0.489 for FA, so it was cheating. Our winner had **ΔFA = +0.117** and **ΔFBTC = +0.069**, both positive.

---

## 9. The whole pipeline in one picture

```
Raw EEG → Preprocessing → Stage 1: detection backbone → "Seizure?"
                                                          ├─ No  → stop
                                                          └─ Yes → Stage 2
Stage 2: [frozen backbone] + [fine-tuned backbone] → 256 numbers
         → SMOTE (training only) → HistGB → FA / FIA / FBTC
```

**Preprocessing** just cleans the data: a standard 18-channel layout, removing power-line hum (50/60 Hz), keeping only 0.5–45 Hz where brain activity lives, a common sampling rate, and removing windows with huge artifacts such as muscle movement.

---

## 10. How to read the results

- **Accuracy:** the fraction of all windows labelled correctly. It is misleading here, because always guessing FIA gives about 90%.
- **Recall (per class):** of the windows that truly were class X, how many we found. FA recall 25% means we found 4 of 16.
- **Macro-F1:** an average score across the three classes that treats the rare ones equally. This is the fair single number.
- **AUROC:** for detection, 1.0 is perfect and 0.5 is a coin flip. Ours is 0.824.
- **Confusion matrix:** a grid where rows are the truth and columns are the model's guess. The diagonal is correct answers.

| Model | Accuracy | Macro-F1 | FA recall | FIA recall | FBTC recall |
|---|---|---|---|---|---|
| 1D-CNN baseline | 0.897 | 0.315 | 0 | 0.99 | 0 |
| Fine-tuned CRNN (v2) | 0.907 | 0.526 | 0 | 0.97 | 0.46 |
| **EEG-HDC v4** | **0.911** | **0.612** | **0.25** | 0.96 | **0.51** |

All three score about 91% accuracy, but the baseline never finds a rare seizure. v4 is the only one that detects all three types, and its macro-F1 roughly doubles.

---

## 11. Honest limitations (examiners will ask)

- The rare classes come from **one patient each**, so the FA test set has only 16 windows.
- v2 and v4 split data by seizure event, so the same patient can appear in train and test. That's an easier test than the baseline's leave-one-patient-out.
- FA recall of 25% is still low, so this is a research result, not a clinical tool.

---

### If you get asked "why this design?"

*"Detection has lots of data and subtypes have little, so we transfer knowledge from detection. Neither backbone alone worked, so we combine both. SMOTE fixes the imbalance in fingerprint space. And we prove the model isn't cheating by testing it on fake-labelled normal EEG."*

If it would help, I can also add short speaker notes to the slides, one per slide, in this simple language.
