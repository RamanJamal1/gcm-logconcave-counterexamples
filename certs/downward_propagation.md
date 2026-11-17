# Downward propagation of a universal entropy sign

Convention: H(p)=integral p log p and P_t=exp(t*d_xx/2). Define S_r(P_t p)=(-1)^r d_t^r H(P_t p).

Assume an exact identity proves S_M(q)>=0 for every log-concave q at positive heat times, with integration by parts justified as in paper/analytic_scope.tex. This implies S_r(P_t p)>=0 for every 1<=r<=M and t>0.

Proof: Gaussian conditional-Hermite identities give, for every finite q>=1,

    integral p_t |p_t^(k)/p_t|^q <= t^(-k*q/2) E|He_k(Z)|^q.

The score derivatives are weighted-homogeneous polynomials in these density ratios. The reduced polynomial for S_r has weighted degree 2r. Holder's inequality therefore gives |S_r(P_t p)|<=C_r*t^(-r). In particular S_r tends to zero at infinity. On t>0, differentiation gives d_t S_r=-S_(r+1), hence

    S_r(P_t p) = integral_t^infinity S_(r+1)(P_s p) ds.

The integral converges for r>=1 by the same bound. Since Gaussian convolution preserves log-concavity, the universal sign at M applies along the entire heat trajectory. Induction gives all lower signs. A right Taylor expansion of H(P_t p) at zero transfers the sign to the corresponding right derivative, through signed forward differences. This is the endpoint interpretation used in the research question and note.

This lemma is analytic, not a numerical search inference. A certificate for a single density, or a family not preserved by heat convolution, does not suffice for its premise.
