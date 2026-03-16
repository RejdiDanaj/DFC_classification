import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sklearn.covariance import graphical_lasso



data_dir = "ica_timeseries"
out_dir = "hmm_mar"
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(data_dir, "*_timeseries.npy")))

subjects = []
subject_ids = []

for f in files:
    S = np.load(f)  # (T, C)
    S = (S - S.mean(axis=0)) / S.std(axis=0)  # z-score per subject
    subjects.append(S)
    subject_ids.append(os.path.basename(f).split("_")[0])

print(f"Loaded {len(subjects)} subjects")
print("Example shape:", subjects[0].shape)


#define the gaussian likelyhood for later use 
def log_gaussian(X, mean, cov):
    # Full multivariate Gaussian log-likelihood
    C = X.shape[1]
    cov += 1e-6 * np.eye(C)
    sign, logdet = np.linalg.slogdet(cov)
    diff = X - mean
    return -0.5 * (C * np.log(2 * np.pi) + logdet + np.sum(diff @ np.linalg.inv(cov) * diff, axis=1))


# Diagonal Gaussian likelihood in case of convergence issues note if the diag only emission is use you inheretly lose connectivity information 
# this funciton is mainly used to test code functionality and not used for dfc construction
def log_gaussian_diag(X, mean, var):
    return -0.5 * (
        np.sum(np.log(2 * np.pi * var)) +
        np.sum((X - mean) ** 2 / var, axis=1)
    )




def forward_backward(log_emlik, logA, logpi):
    T, K = log_emlik.shape
    log_alpha = np.zeros((T, K))
    log_beta = np.zeros((T, K))

    log_alpha[0] = logpi + log_emlik[0]

    for t in range(1, T):
        log_alpha[t] = log_emlik[t] + logsumexp(
            log_alpha[t - 1][:, None] + logA, axis=0
        )

    log_beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(
            logA + log_emlik[t + 1] + log_beta[t + 1], axis=1
        )

    loglik = logsumexp(log_alpha[-1])
    gamma = np.exp(log_alpha + log_beta - loglik)

    return gamma, loglik


# Group-level HMM
def fit_group_hmm(subjects, K=5, n_iter=50, eps=1e-6):

    X = np.vstack(subjects)
    T, C = X.shape
    # Initialisation
    pi = np.ones(K) / K
    A = np.ones((K, K)) / K
    rng = np.random.default_rng(0)
    means = X[rng.choice(T, K, replace=False)]
    vars_ = np.ones((K, C))

    for it in range(n_iter):
        #  E STEP 
        log_emlik = np.column_stack([
            log_gaussian(X, means[k], vars_[k])
            for k in range(K)
        ])

        gamma, loglik = forward_backward(
            log_emlik,
            np.log(A + eps),
            np.log(pi + eps)
        )
        Nk = gamma.sum(axis=0)

        #  M-STEP 
        pi = Nk / Nk.sum()
        for k in range(K):
            # Update means
            means[k] = (gamma[:, k][:, None] * X).sum(axis=0) / Nk[k]
            diff = X - means[k]
            # Weighted covariance
            weighted = gamma[:, k][:, None] * diff
            emp_cov = (weighted.T @ diff) / Nk[k]
            # Graphical LASSO regularization from sklearn
            cov_reg, _ = graphical_lasso(
                emp_cov + np.eye(C) * eps,
                alpha=0.02,
                max_iter=200
            )
            # store the full states
            vars_[k] = cov_reg + eps * np.eye(C) #store it as an array 
            #note if convergence is probelmatic use only the diagonal 
            #vars_[k] = np.diag(cov_reg) + eps
        # Transition update 
        xi_sum = np.zeros((K, K))
        idx = 0
        for S in subjects:
            T_s = S.shape[0]
            log_emlik_s = log_emlik[idx:idx + T_s]

            gamma_s, _ = forward_backward(
                log_emlik_s,
                np.log(A + eps),
                np.log(pi + eps)
            )

            for t in range(T_s - 1):
                xi = np.exp(
                    gamma_s[t][:, None]
                    + np.log(A + eps)
                    + log_emlik_s[t + 1]
                    - logsumexp(gamma_s[t])
                )
                xi_sum += xi
            idx += T_s
        A = xi_sum / xi_sum.sum(axis=1, keepdims=True)

        if it % 10 == 0:
            print(f"Iter {it}, loglik ≈ {loglik:.2f}")

    return pi, A, means, vars_

K = 5
pi_g, A_g, means_g, vars_g = fit_group_hmm(
    subjects,
    K=K,
    n_iter=50
)



# Infer gamma for individual subjects
def infer_subject_gamma(S, pi, A, means, vars_):
    log_emlik = np.column_stack([
        log_gaussian(S, means[k], vars_[k])
        for k in range(len(means))
    ])
    gamma, _ = forward_backward(
        log_emlik,
        np.log(A + 1e-12),
        np.log(pi + 1e-12)
    )
    return gamma



# Continuous DFC construction for individuals
def continuous_covariance(gamma, means, vars_):

    T, K = gamma.shape
    C = means.shape[1]

    covs = np.zeros((K, C, C))

    for k in range(K):
        covs[k] = vars_[k]

    mu_t = gamma @ means
    mean_outer = np.einsum("ki,kj->kij", means, means)

    weighted_cov = np.tensordot(gamma, covs, axes=(1, 0))
    weighted_mean = np.tensordot(gamma, mean_outer, axes=(1, 0))
    mu_outer = np.einsum("ti,tj->tij", mu_t, mu_t)

    return weighted_cov + weighted_mean - mu_outer



def graphical_lasso_time(cov_time, alpha=0.05):

    T, C, _ = cov_time.shape
    precisions = np.zeros_like(cov_time)

    for t in range(T):
        cov = cov_time[t] + np.eye(C) * 1e-6
        _, prec = graphical_lasso(cov, alpha=alpha, max_iter=200)
        precisions[t] = prec

    return precisions



# Example visualization
rng = np.random.default_rng(1)
idx = rng.integers(len(subjects))

S = subjects[idx]
gamma = infer_subject_gamma(S, pi_g, A_g, means_g, vars_g)

plt.figure(figsize=(12, 4))
plt.imshow(gamma.T, aspect="auto", origin="lower", cmap="viridis")
plt.xlabel("Time")
plt.ylabel("State")
plt.title(f"State probabilities – {subject_ids[idx]}")
plt.colorbar()
plt.show()


plt.figure(figsize=(6, 4))
plt.bar(range(K), gamma.mean(axis=0))
plt.xlabel("State")
plt.ylabel("Fractional occupancy")
plt.show()



def transition_entropy(A):
    A = A / A.sum(axis=1, keepdims=True)
    return -np.sum(A * np.log(A + 1e-12), axis=1).mean()



def emission_mutual_information(gamma):

    p_k = gamma.mean(axis=0)
    H_Z = -np.sum(p_k * np.log(p_k + 1e-12))

    H_Z_t = -np.sum(gamma * np.log(gamma + 1e-12), axis=1)

    return H_Z - H_Z_t.mean()



def individual_transition_matrix(gamma):

    T, K = gamma.shape
    A = np.zeros((K, K))

    for t in range(T - 1):
        A += np.outer(gamma[t], gamma[t + 1])

    return A / A.sum(axis=1, keepdims=True)



# Sanity check
idx = np.random.randint(len(subjects))
sid = subject_ids[idx]
S = subjects[idx]

gamma = infer_subject_gamma(S, pi_g, A_g, means_g, vars_g)
A_ind = individual_transition_matrix(gamma)

H_trans = transition_entropy(A_ind)
MI_emit = emission_mutual_information(gamma)

print("Subject:", sid)
print("-" * 40)

print("Gamma shape:", gamma.shape)
print("Gamma min / max:", gamma.min(), gamma.max())

print("Row sum (min / max):",
      gamma.sum(axis=1).min(),
      gamma.sum(axis=1).max())
print("Mean state occupancies:", gamma.mean(axis=0))
print("Occupancy sum:", gamma.mean(axis=0).sum())
print("\nIndividual transition matrix A:")
print(A_ind)
print("Row sums of A:", A_ind.sum(axis=1))

print("\n transition entropy:")
print("H =", H_trans)

print("\n mutual information:")
print("MI =", MI_emit)

cov_time = continuous_covariance(gamma, means_g, vars_g)
eigvals = np.linalg.eigvalsh(cov_time[cov_time.shape[0] // 2])
print("\nCovariance eigenvalues:")
print("min eig:", eigvals.min())
print("max eig:", eigvals.max())



# Save subject results
out_dir = "hmm_mar"
os.makedirs(out_dir, exist_ok=True)

for sid, S in zip(subject_ids, subjects):
    gamma = infer_subject_gamma(S, pi_g, A_g, means_g, vars_g)
    cov_time = continuous_covariance(gamma, means_g, vars_g)
    A_ind = individual_transition_matrix(gamma)
    H_trans = transition_entropy(A_ind)
    MI_emit = emission_mutual_information(gamma)
    np.savez(
        os.path.join(out_dir, f"{sid}_hmm_results.npz"),
        gamma=gamma,
        cov_time=cov_time,
        A_ind=A_ind,
        transition_entropy=H_trans,
        emission_mutual_info=MI_emit
    )