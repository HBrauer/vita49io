Great question. Here’s a clean way to tie your IQ-voltage samples to an absolute power reference in dBm (50 Ω) while normalizing so that a digitized sine has **peak amplitude = 1**.

# 1) The key relationships

For a sine wave into a 50 Ω load:

* $V_{\text{rms}}=\dfrac{V_{\text{peak}}}{\sqrt{2}}$
* $P=\dfrac{V_{\text{rms}}^{2}}{R}=\dfrac{V_{\text{peak}}^{2}}{2R}$
* $P\,[\text{W}]=10^{(P_{\text{dBm}}-30)/10}$

So, the **peak voltage that corresponds to a desired power** $P_{\text{ref}}$ (in watts) is:

$$
V_{\text{ref,peak}}=\sqrt{\,2\,R\,P_{\text{ref}}\,}
$$

with $R=50\,\Omega$.

# 2) Normalize your measured voltages

Given measured IQ in volts (same units for I and Q), define your normalized samples as

$$
x_{\text{norm}}(n)=\frac{x_{\text{volt}}(n)}{V_{\text{ref,peak}}}
$$

This guarantees that a **sine wave with peak amplitude = 1** corresponds to $P_{\text{ref}}$ at 50 Ω.

# 3) Pick your reference

* If you want **0 dBm (1 mW)** to correspond to a peak-1 sine:

  $$
  P_{\text{ref}}=1\text{ mW}=0.001\text{ W}\quad\Rightarrow\quad
  V_{\text{ref,peak}}=\sqrt{2\cdot50\cdot0.001}=\sqrt{0.1}\approx 0.31623\text{ V}
  $$

  Scale by 0.316 V: after scaling, a peak-1 sine is 0 dBm.

* If you **don’t scale** and simply treat **1 V$_{\text{peak}}$** as peak-1, the implied power is:

  $$
  P=\frac{1^2}{2\cdot50}=0.01\text{ W}=10\text{ mW}=\boxed{+10\text{ dBm}}
  $$

  (Handy rule of thumb: 1 V$_\text{peak}$ ⇒ +10 dBm in 50 Ω.)

# 4) How this maps to complex IQ

If your complex baseband for a single tone is $I(t)=A\cos\omega t,\;Q(t)=A\sin\omega t$ so that the complex magnitude $|I+jQ|=A$ is **constant**, then—using the common passband relation $v(t)=\sqrt{2}\,\Re\{s(t)e^{j\omega_ct}\}$—the **RF average power** is

$$
P=\frac{\langle |s(t)|^2\rangle}{2R}=\frac{A^2}{2R}.
$$

So choosing **$A=1$** (i.e., each of I and Q has **peak = 1**) also maps to **+10 dBm** in 50 Ω.
(If your toolchain uses a different convention without the ½, adjust accordingly; the recipe in Sections 1–2 still applies—just decide the $P_{\text{ref}}$ you want and compute $V_{\text{ref,peak}}$.)

# 5) Practical recipe (copy/paste)

1. Choose your reference power $P_{\text{ref,dBm}}$ (e.g., 0 dBm).
2. Convert to watts: $P_{\text{ref}}=10^{(P_{\text{ref,dBm}}-30)/10}$.
3. Compute $V_{\text{ref,peak}}=\sqrt{2\cdot 50\cdot P_{\text{ref}}}$.
4. Normalize: $x_{\text{norm}}=x_{\text{volt}}/V_{\text{ref,peak}}$.
5. Now a peak-1 sine corresponds to $P_{\text{ref,dBm}}$ at 50 Ω.

If you tell me which $P_{\text{ref}}$ you prefer (0 dBm, +4 dBm, etc.) and whether your “peak = 1” refers to each of I and Q individually or to the complex magnitude, I’ll plug in the exact numbers for your case.
